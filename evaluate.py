import os
import argparse
import albumentations as A

import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from data import Dataset
from utils import dice_coeff

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval_on_test_set(dataset_loc: str = None,
                     model_path: str = None,
                     batch_size: int = 32,
                     num_workers: int = 1) -> None:
    
    test_images = os.path.join(dataset_loc, "images/val")
    test_masks = os.path.join(dataset_loc, "masks/val")
    list_of_test_images = os.listdir(test_images)
    
    test_transform = A.Compose([A.Resize(512, 512)])

    train_dataset = Dataset(images=list_of_test_images,
                            image_folder=test_images,
                            mask_folder=test_masks,
                            transform=test_transform)

    test_dataloader = DataLoader(train_dataset,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 shuffle=False)


    model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b4",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1)
    
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    iou = 0.0
    acc = 0.0
    num_batches = 0
    
    with torch.inference_mode():
        for images, masks in test_dataloader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            masks = masks.to(torch.uint8)
            tp, fp, fn, tn = smp.metrics.get_stats(outputs, masks, mode='binary', threshold=0.5)
            iou += smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            acc += smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
            num_batches += 1
    avg_iou = iou / num_batches
    avg_acc = acc / num_batches
    print(f"Average IoU Coefficient on Test Set: {avg_iou:.4f}")
    print(f"Average Accuracy on Test Set: {avg_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_loc", type=str, default=None, help="Path to the dataset to train.")
    parser.add_argument("-m", "--model_path", default=None, help="Path to saved model to evaluate.")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size of the dataset to train.")
    parser.add_argument("-w", "--num_workers", type=int, default=1, help="Number of workers to use for training.")
    args = parser.parse_args()

    eval_on_test_set(
        dataset_loc=args.dataset_loc,
        model_path=args.model_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )