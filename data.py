import os
import torch
import numpy as np
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    def __init__(self, images, image_folder, ndvi_folder, mask_folder, transform=None):
        self.images = images
        self.image_folder = image_folder
        self.ndvi_folder = ndvi_folder
        self.mask_folder = mask_folder
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.image_folder, image_name)
        
        mask_name = os.path.splitext(image_name)[0] + '.png'
        mask_path = os.path.join(self.mask_folder, mask_name)
        
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0
        
        ndvi_path = os.path.join(self.ndvi_folder, image_name)
        ndvi_image = Image.open(ndvi_path)
        ndvi_image = np.array(ndvi_image, dtype=np.float32)
        combined_image = np.dstack((image, ndvi_image))
        if self.transform:
            augmented = self.transform(image=combined_image, mask=mask)
            combined_image = augmented['image']
            mask = augmented['mask']
        return torch.from_numpy(combined_image).permute(2, 0, 1), torch.from_numpy(mask).unsqueeze(0)