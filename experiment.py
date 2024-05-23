import os
import mlflow
import argparse
from tqdm import tqdm
from art import tprint
from typing import Tuple
import albumentations as A

import torch
from torch import nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from model import UNet
from data import Dataset
from utils import DICE_BCE_Loss, dice_coeff, set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

def train_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               loss_fn: DICE_BCE_Loss,
               device: torch.device) -> Tuple[float, float]:
    model.train()
    loss = 0.0
    dice = 0.0
    for i, (images, masks) in enumerate(dataloader):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        z = model(images)
        J = loss_fn(z, masks)
        J.backward()
        optimizer.step()
        loss += J.item()
        dice += dice_coeff(z, masks).item()
    loss = loss / len(dataloader)
    dice = dice / len(dataloader)
    return loss, dice

def eval_step(model: nn.Module,
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: DICE_BCE_Loss,
              device: torch.device) -> Tuple[float, float]:
    model.eval()
    loss = 0.0
    dice = 0.0
    with torch.inference_mode():
        for i, (images, masks) in enumerate(dataloader):
            images, masks = images.to(device), masks.to(device)
            z = model(images)
            J = loss_fn(z, masks)
            loss += J.item()
            dice += dice_coeff(z, masks).item()
    loss = loss / len(dataloader)
    dice = dice / len(dataloader)
    return loss, dice

def train_loop(dataset_loc: str = None,
               num_epochs: int = 1,
               batch_size: int = 32,
               num_workers: int = 1,
               model_path: str = None) -> None:

    set_seed(seed=SEED)

    train_images = os.path.join(dataset_loc, "images/train")
    train_masks = os.path.join(dataset_loc, "masks/train")
    list_of_train_images = os.listdir(train_images)

    val_images = os.path.join(dataset_loc, "images/val")
    val_masks = os.path.join(dataset_loc, "masks/val")
    list_of_val_images = os.listdir(val_images)

    train_transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
    ])

    val_transform = A.Compose([
        A.Resize(256, 256),
    ])

    train_dataset = Dataset(images=list_of_train_images,
                            image_folder=train_images,
                            mask_folder=train_masks,
                            transform=train_transform)

    val_dataset = Dataset(images=list_of_val_images,
                          image_folder=val_images,
                          mask_folder=val_masks,
                          transform=val_transform)
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=True)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers, 
                                shuffle=False)
    
    model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    model.to(device)

    run_name = f"{os.path.splitext(os.path.basename(model_path))[0]}_{num_epochs}_epochs"
    with mlflow.start_run(run_name=run_name):
        best_val_loss = float('inf')
        best_model_state_dict = None
        for epoch in tqdm(range(num_epochs)):
            train_loss, train_dice = train_step(model=model, 
                                                dataloader=train_dataloader,
                                                optimizer=optimizer,
                                                loss_fn=loss_fn,
                                                device=device)

            val_loss, val_dice = eval_step(model=model, 
                                           dataloader=val_dataloader,
                                           loss_fn=loss_fn,
                                           device=device)    
            print(
                f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_dice: {train_dice:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"val_dice: {val_dice:.4f} | "
            )

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_dice", train_dice, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_dice", val_dice, step=epoch)

            scheduler.step(val_loss)

            artifact_uri = mlflow.get_artifact_uri()
            artifact_uri = artifact_uri.split("file://")[-1]
            artifact_model_path = os.path.join(artifact_uri, model_path.split("/")[-1])
            torch.save(model.state_dict(), artifact_model_path)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state_dict = model.state_dict()

        if best_model_state_dict is not None:
            if model_path.endswith(".pth") or model_path.endswith(".pt"):
                torch.save(best_model_state_dict, model_path)
            else:
                torch.save(best_model_state_dict, model_path + ".pth")
            print(f"Best validation loss: {best_val_loss:.4f}")
        print("DONE")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_loc", type=str, default=None, help="Path to the dataset to train.")
    parser.add_argument("-e", "--num_epochs", type=int, default=1, help="Number of epochs to train for.")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size of the dataset to train.")
    parser.add_argument("-w", "--num_workers", type=int, default=1, help="Number of workers to use for training.")
    parser.add_argument("-m", "--model_path", default=None, help="Path to save model to.")
    args = parser.parse_args()

    train_loop(
        dataset_loc=args.dataset_loc,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model_path=args.model_path,
    )