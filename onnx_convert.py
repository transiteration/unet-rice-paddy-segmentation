import onnx
import torch
import argparse
import onnxruntime
import numpy as np
from model import UNet
import segmentation_models_pytorch as smp

def pth_to_onnx(model_path: str = None, onnx_path: str = None) -> None:
    model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b4",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1)
    
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    dummy_input = torch.randn(1, 3, 512, 512)
    torch.onnx.export(model,
                      dummy_input,
                      onnx_path,
                      verbose=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", default=None, help="Path to saved PyTorch model.")
    parser.add_argument("-o", "--onnx_path", default=None, help="Path to where save onnx model.")
    args = parser.parse_args()

    pth_to_onnx(model_path=args.model_path,
                onnx_path=args.onnx_path)




