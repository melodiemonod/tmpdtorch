from abc import ABC, abstractmethod
from sympy import denom
import torch
import numpy as np

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
        
        sigma_y = measurement.std().item() ** 2

        def A(v, x):
            Ht_v = self.operator.transpose(v, **kwargs)      # H^T v
            X_Ht_v = x * Ht_v                 # X H^T v (diagonal X)
            HXHt_v = self.operator.forward(X_Ht_v, **kwargs) # H X H^T v
            return HXHt_v + sigma_y * v       # (H X H^T + σI)v

        def cg(A, b, x, iters=10):
            z = torch.zeros_like(b)

            r = b - A(z, x)
            p = r.clone()
            rs_old = (r * r).sum()

            for _ in range(iters):
                Ap = A(p, x)

                denom = (p * Ap).sum() + 1e-8
                alpha = rs_old / denom

                z = z + alpha * p
                r = r - alpha * Ap

                rs_new = (r * r).sum()

                if rs_new < 1e-8:
                    break

                p = r + (rs_new / rs_old) * p
                rs_old = rs_new

            return z

        # ∇xnm0|t
        grad = torch.autograd.grad(outputs=x_0_hat, grad_outputs=torch.ones_like(x_0_hat), inputs=x_prev)[0]

        # y − Hm0|t
        yadj = measurement - self.operator.forward(x_0_hat, **kwargs)
        
        # part_2 = (v_n / sqrt(alpha_n) H ∇xnm0|t H^T + sigma_y^2 I)^-1 (y − Hm0|t)
        sqrt_alpha_n = np.sqrt(1 - v_n)
        part_2 = cg(A, yadj, v_n / sqrt_alpha_n * grad, iters=10)

        # part 1 = (v_n / sqrt(alpha_n) H ∇xnm0|t H^T part_2
        part_1 = v_n / sqrt_alpha_n * grad * self.operator.transpose(part_2, **kwargs)
        
        # pred_xstart_y = m0|t + part_1
        pred_xstart_y = x_0_hat + part_1
        
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
