import os
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms.functional as TF
from torchvision import transforms
import random

import os
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import functools

import os
import functools
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class GenericNpyDataset(Dataset):
    def __init__(self, condition_directory: str, original_directory: str, transform=None, test_flag: bool = False):
        super().__init__()
        self.condition_directory = os.path.expanduser(condition_directory)
        self.original_directory = os.path.expanduser(condition_directory)
        self.test_flag = test_flag
        
        # Precompute full file paths
        self.image_files = sorted([os.path.join(self.image_directory, x) for x in os.listdir(self.image_directory) if x.endswith('.npy')])
        self.mask_files = sorted([os.path.join(self.mask_directory, x) for x in os.listdir(self.mask_directory) if x.endswith('.npy')])
        
        if transform is None:
            self.transform = self.get_default_transform()
        else:
            self.transform = transform

    def get_default_transform(self):
        return A.Compose([
            ToTensorV2(),
        ])

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx: int):
        # Load and preprocess image
        image = np.load(self.image_files[idx], mmap_mode='r')
        image = np.array(image).copy()  # Ensure the array is writable
        image = image.transpose((1, 2, 0))
        
        # Load and preprocess mask
        mask = np.load(self.mask_files[idx], mmap_mode='r')
        mask = np.array(mask).copy()  # Ensure the array is writable
        mask = mask.transpose((1, 2, 0))
    
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
            # Convert back to width x height x channel only if the channel dimension is 1
            if mask.shape[2] == 1:
                mask = mask.permute(2, 0, 1)
        
        return image, mask

    def __len__(self):
        return len(self.image_files)
    




    

    



    
