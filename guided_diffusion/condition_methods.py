from abc import ABC, abstractmethod
import torch
import numpy as np
import functorch

__CONDITIONING_METHOD__ = {}

def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)

    
class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser
    
    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)
    
    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == 'gaussian':
            grad = torch.autograd.grad(outputs=x_0_hat, inputs=x_prev)[0]
        
        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement-Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError
             
        return grad
   
    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass
    
@register_conditioning_method(name='vanilla')
class Identity(ConditioningMethod):
    # just pass the input without conditioning
    def conditioning(self, x_t):
        return x_t
    
@register_conditioning_method(name='projection')
class Projection(ConditioningMethod):
    def conditioning(self, x_t, noisy_measurement, **kwargs):
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement)
        return x_t


@register_conditioning_method(name='mcg')
class ManifoldConstraintGradient(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        
    def conditioning(self, x_prev, x_t, x_0_hat, measurement, noisy_measurement, **kwargs):
        # posterior sampling
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        
        # projection
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement, **kwargs)
        return x_t, norm
        
@register_conditioning_method(name='ps')
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, v_n, q_posterior_mean, x_0_hat, measurement, **kwargs):
        
        sigma_y = 0.05 ** 2
        sqrt_alpha_n = np.sqrt(1.0 - v_n)
        scale = 0.5

        # diagonal Jacobian approximation (∇x_t m0|t)
        J_diag = torch.autograd.grad(
            outputs=x_0_hat,
            inputs=x_prev,
            grad_outputs=torch.ones_like(x_0_hat)
        )[0]

        # # residual: y − Hm0|t
        yadj = measurement - self.operator.forward(x_0_hat, **kwargs)

        # approximate denominator (NO CG)
        Ht_one = self.operator.transpose(torch.ones_like(yadj), **kwargs)
        HdHt = self.operator.forward(J_diag * Ht_one, **kwargs)
        denom = (v_n / sqrt_alpha_n) * HdHt + sigma_y

        # inverse approximation
        z = yadj / (denom + 1e-8)
        
        # final correction
        part_1 = (v_n / sqrt_alpha_n) * J_diag * self.operator.transpose(z, **kwargs)
                
        pred_xstart_y = x_0_hat + scale * part_1
        
        # predict with ddpm
        img = q_posterior_mean(x_t = x_prev, x_start = pred_xstart_y)
        
        # calculate distance
        distance = torch.linalg.norm(measurement - self.operator.forward(pred_xstart_y, **kwargs)) 
        
        return img, distance
        
@register_conditioning_method(name='ps+')
class PosteriorSamplingPlus(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.num_sampling = kwargs.get('num_sampling', 5)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm = 0
        for _ in range(self.num_sampling):
            # TODO: use noiser?
            x_0_hat_noise = x_0_hat + 0.05 * torch.rand_like(x_0_hat)
            difference = measurement - self.operator.forward(x_0_hat_noise)
            norm += torch.linalg.norm(difference) / self.num_sampling
        
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        x_t -= norm_grad * self.scale
        return x_t, norm


@register_conditioning_method(name="tmp")
class TweedieMomentProjection(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.num_sampling = kwargs.get("num_sampling", 5)
        # self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_t, measurement, estimate_x_0, r, v, noise_std, **kwargs):
        def estimate_h_x_0(x_t):
            x_0, model_var_values = estimate_x_0(x_t)
            return self.operator.forward(x_0, **kwargs), (x_0, model_var_values)
            # return self.operator.forward(x_0, **kwargs)

        if self.noiser.__name__ == "gaussian":
            # Due to the structure of this code, the condition operator is not accesible unless inside from in the conditioning method. That's why the analysis is here
            # Since functorch 1.1.1 is not compatible with this
            # functorch 0.1.1 (unstable; works with PyTorch 1.11) does not work with autograd.Function, which is what the model is written in. It can be rewritten, or package environment needs to be solved.
            # h_x_0, vjp = torch.autograd.functional.vjp(estimate_h_x_0, x_t, self.operator.forward(torch.ones_like(x_t), **kwargs))
            # difference = measurement - h_x_0
            # norm = torch.linalg.norm(difference)
            # C_yy = self.operator.forward(vjp, **kwargs) + noise_std**2 / ratio
            # _, ls = torch.autograd.functional.vjp(estimate_h_x_0, x_t, difference / C_yy)
            # x_0 = estimate_x_0(x_t)

            # NOTE: This standing functorch way seems to be only slightly faster (163 seconds instead of 188 seconds)
            # NOTE: In torch, usually our method is up to 2x slower than dps due to the extra vjp
            # # h_x_0, vjp_estimate_h_x_0, x_0 = torch.func.vjp(estimate_h_x_0, x_t, has_aux=True)
            h_x_0, vjp_estimate_h_x_0, (x_0, model_var_values) = functorch.vjp(
                estimate_h_x_0, x_t, has_aux=True
            )
            C_yy = (
                self.operator.forward(
                    vjp_estimate_h_x_0(
                        self.operator.forward(torch.ones_like(x_0), **kwargs)
                    )[0],
                    **kwargs,
                )
                + noise_std**2 / r
            )
            difference = measurement - h_x_0
            norm = torch.linalg.norm(difference)
            ls = vjp_estimate_h_x_0(difference / C_yy)[0]
            del C_yy

            x_0 = x_0 + ls  # commenting it out shows that rest of the code works
            del ls
        else:
            raise NotImplementedError

        return x_0, norm, model_var_values