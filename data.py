import os
import torch
import numpy as np
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    def __init__(self, images, image_folder, mask_folder, transform=None):
        self.images = images
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.images[idx])
        mask_path = os.path.join(self.mask_folder, self.images[idx])
        
        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")
        
        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        return torch.from_numpy(image).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0)