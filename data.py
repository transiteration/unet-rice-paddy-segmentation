import os
import cv2
import torch
from PIL import Image
import numpy as np

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
        
        image = Image.open(image_path)
        image = image.convert("RGB")
        image = np.asarray(image)
        
        mask = Image.open(mask_path)
        mask = mask.convert("L")
        mask = np.asarray(mask)
        
        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        return torch.from_numpy(image).permute(2, 0, 1), torch.from_numpy(mask).unsqueeze(0)
    

