from glob import glob
from PIL import Image
from typing import Callable, Optional
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
import random

__DATASET__ = {}

def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls
    return wrapper


def get_dataset(name: str, root: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    return __DATASET__[name](root=root, **kwargs)


def get_dataloader(dataset: VisionDataset,
                   batch_size: int, 
                   num_workers: int, 
                   train: bool):
    dataloader = DataLoader(dataset, 
                            batch_size, 
                            shuffle=train, 
                            num_workers=num_workers, 
                            drop_last=train)
    return dataloader


@register_dataset(name='ffhq')
class FFHQDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None, num_samples: Optional[int] = None, seed: int = 0,):
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + '/**/*.png', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."
        
        # deterministic random sampling
        if num_samples is not None:
            rng = random.Random(seed)
            self.fpaths = rng.sample(self.fpaths, num_samples)

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img
    
@register_dataset(name='imagenet')
class ImageNetDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None, num_samples: Optional[int] = None, seed: int = 0,):
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + '/**/*.JPEG', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."
        
        # deterministic random sampling
        if num_samples is not None:
            rng = random.Random(seed)
            self.fpaths = rng.sample(self.fpaths, num_samples)

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img