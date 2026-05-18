from functools import partial
import os
import argparse
import time
from PIL import Image
import yaml
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config', type=str)
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    args = parser.parse_args()
    
    print("save_dir: ", args.save_dir)
   
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
    data_config = load_yaml(args.data_config)
   
    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    operator.seed = 42
    noiser = get_noise(**measure_config['noise'])
    noiser.seed = 42
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
   
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)
   
    # Working directory
    out_path = args.save_dir
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    transform = transforms.Compose([
            transforms.Resize(256, interpolation=Image.BICUBIC),
            transforms.CenterCrop(256),   # optional but commonly included
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                [0.5, 0.5, 0.5]),
        ])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # Exception) In case of inpainting, we need to generate a mask 
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
           **measure_config['mask_opt']
        )
        
    # Do Inference
    for i, ref_img in enumerate(loader):

        print(f"Processing image {i} / {len(loader)}")
        
        # check if the image is already processed
        fname = str(i).zfill(5) + '.npy'
        if os.path.exists(os.path.join(out_path, 'input', fname)):
            logger.info(f"Image {i} already exists. Skipping...")
            continue
        
        start_time = time.time()
        logger.info(f"Inference for image {i}")
        ref_img = ref_img.to(device)
        operator.seed += 1
        noiser.seed += 1

        # Exception) In case of inpainging,
        if measure_config['operator'] ['name'] == 'inpainting':
            mask = mask_gen(ref_img, seed=operator.seed)
            mask = mask[:, 0, :, :].unsqueeze(dim=0)
            measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
            sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)

            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img, mask=mask)
            y_n = noiser(y)

        else: 
            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img)
            y_n = noiser(y)
         
        # Sampling
        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        sample = sample_fn(x_start=x_start, measurement=y_n, record=True, save_root=out_path)

        input_arr = clear_color(y_n)
        label_arr = clear_color(ref_img)
        recon_arr = clear_color(sample)

        # save numpy arrays
        fname = str(i).zfill(5) + '.npy'
        np.save(os.path.join(out_path, 'input', fname), input_arr)
        np.save(os.path.join(out_path, 'label', fname), label_arr)
        np.save(os.path.join(out_path, 'recon', fname), recon_arr)

        # save images
        fname = str(i).zfill(5) + '.png'
        plt.imsave(os.path.join(out_path, 'input', fname), input_arr)
        plt.imsave(os.path.join(out_path, 'label', fname), label_arr)
        plt.imsave(os.path.join(out_path, 'recon', fname), recon_arr)

        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Time per image {i}: {elapsed:.2f} sec")
        
        if i >= data_config['num_samples'] - 1:
            break
    
if __name__ == '__main__':
    main()
