import os
import argparse
from pathlib import Path
import albumentations as A

import torch
from model import UNet
from data import Dataset
from utils import dice_coeff

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval_on_test_set(dataset_loc: str = None,
                     model_path: str = None,
                     batch_size: int = 32,
                     num_workers: int = 1) -> None:
    
    test_images = Path(dataset_loc, "test/images")
    test_masks = Path(dataset_loc, "test/masks")
    list_of_test_images = os.listdir(test_images)
    
    test_transform = A.Compose([A.Resize(256, 256)])

    train_dataset = Dataset(images=list_of_test_images,
                        image_folder=test_images,
                        mask_folder=test_masks,
                        transform=test_transform)

    test_dataloader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            shuffle=True)


    model = UNet(1, 1)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    total_dice_coeff = 0.0
    num_batches = 0
    
    with torch.inference_mode():
        for images, masks in test_dataloader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            batch_dice_coeff = dice_coeff(outputs, masks)
            total_dice_coeff += batch_dice_coeff
            num_batches += 1
    avg_dice_coeff = total_dice_coeff / num_batches
    print(f"Average Dice Coefficient on Test Set: {avg_dice_coeff:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_loc", type=str, default=None, help="Path to the dataset to train.")
    parser.add_argument("--model_path", default=None, help="Path to saved model to evaluate.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size of the dataset to train.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers to use for training.")
    args = parser.parse_args()

    eval_on_test_set(
        dataset_loc=args.dataset_loc,
        model_path=args.model_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )