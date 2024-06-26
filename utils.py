import os
import torch
import random
import numpy as np
from PIL import Image

ndvis = "dataset/ndvis"
images = "dataset/images"
masks = "dataset/masks"

image_train = os.path.join(images, "train")
image_val = os.path.join(images, "val")
ndvi_train = os.path.join(ndvis, "train")
ndvi_val = os.path.join(ndvis, "val")
mask_train = os.path.join(masks, "train")
mask_val = os.path.join(masks, "val")

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_dim(ndvi_dir: str, 
              image_dir: str,
              mask_dir: str):
    list_of_images = os.listdir(ndvi_dir)
    for filename in list_of_images:
        ndvi_path = os.path.join(ndvi_dir, filename)
        ndvi_image = Image.open(ndvi_path)
        ndvi_image = np.array(ndvi_image, dtype=np.float32)

        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path)
        image = np.array(image, dtype=np.float32)
        
        mask_path = os.path.join(mask_dir, filename.replace(".tif", ".png"))

        ndvi_shape = ndvi_image.shape[:2]
        rgb_shape = image.shape[:2]

        if ndvi_shape != rgb_shape:
            print(f"{filename} shapes are not equal. NDVI shape: {ndvi_shape}, Image shape: {rgb_shape}")
            os.remove(ndvi_path)
            os.remove(image_path)
            os.remove(mask_path)        