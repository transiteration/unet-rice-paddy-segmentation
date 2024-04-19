import os
import mlflow
import argparse
from tqdm import tqdm
from art import tprint
from pathlib import Path
import albumentations as A

import torch
from torch import nn
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import UNet
from data import Dataset
from utils import DICE_BCE_Loss, dice_coeff

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

def train_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               loss_fn: DICE_BCE_Loss,
               device: torch.device) -> float:
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
        dice += dice_coeff(z, masks)
    loss = loss / len(dataloader)
    dice = dice / len(dataloader)
    return loss, dice

def eval_step(model: nn.Module,
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: DICE_BCE_Loss,
              device: torch.device) -> float:
    model.eval()
    loss = 0.0
    dice = 0.0
    with torch.inference_mode():
        for i, (images, masks) in enumerate(dataloader):
            images, masks = images.to(device), masks.to(device)
            z = model(images)
            J = loss_fn(z, masks)
            loss += J.item()
            dice += dice_coeff(z, masks)
    loss = loss / len(dataloader)
    dice = dice / len(dataloader)
    return loss, dice

def train_loop(dataset_loc: str = None,
               num_epochs: int = 1,
               batch_size: int = 32,
               num_workers: int = 1,
               model_path: str = None) -> None:


    train_images = Path(dataset_loc, "train/images")
    train_masks = Path(dataset_loc, "train/masks")
    list_of_train_images = os.listdir(train_images)

    val_images = Path(dataset_loc, "val/images")
    val_masks = Path(dataset_loc, "val/masks")
    list_of_val_images = os.listdir(val_images)
    # train_images = os.listdir(images)

    # train_images, val_images = train_test_split(list_of_images, test_size=0.1, random_state=SEED)

    train_transform = A.Compose([A.Resize(256, 256), 
                             A.HorizontalFlip(p=0.5), 
                             A.VerticalFlip(p=0.5), 
                             A.RandomRotate90(p=0.5)])

    val_transform = A.Compose([A.Resize(256, 256)])

    train_dataset = Dataset(images=list_of_train_images,
                            image_folder=train_images,
                            mask_folder=train_masks,
                            transform=train_transform)

    val_dataset = Dataset(images=list_of_val_images,
                          image_folder=val_images,
                          mask_folder=val_masks,
                          transform=val_transform)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True)

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_workers, 
                                                shuffle=True)
    
    model = UNet(3, 1)
    loss_fn = DICE_BCE_Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)
    model.to(device)

    best_val_loss = float('inf')
    best_model_state_dict = None
    with mlflow.start_run() as run:
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
            f"val_dice: {val_dice:.4f} | ")
            
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_dice", train_dice, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_dice", val_dice, step=epoch)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state_dict = model.state_dict()

        if model_path.endswith(".pth") or model_path.endswith(".pt"):
            torch.save(best_model_state_dict, model_path)
            print(best_val_loss)
        else:
            torch.save(best_model_state_dict, model_path + ".pth")
            print(best_val_loss)
            
        mlflow.log_artifact(model_path)
        tprint("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_loc", type=str, default=None, help="Path to the dataset to train.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs to train for.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size of the dataset to train.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers to use for training.")
    parser.add_argument("--model_path", default=None, help="Path to save model to.")
    args = parser.parse_args()

    train_loop(
        dataset_loc=args.dataset_loc,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model_path=args.model_path,
    )