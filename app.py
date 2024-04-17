import torch
import argparse
import numpy as np
import gradio as gr
from PIL import Image
import albumentations as A
from model import UNet

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default=None, help="Path to saved model.")
args = parser.parse_args()

model = UNet(3, 1)
state_dict = torch.load(args.model_path)
model.load_state_dict(state_dict)

def predict(image_path):
    device = "cpu"
    image = Image.open(image_path)
    image = image.convert("RGB")
    image = np.asarray(image)
    image = np.array(image, dtype=np.float32) / 255.0

    test_transform = A.Compose([A.Resize(256, 256)])
    transformed = test_transform(image=image)

    transformed_image = transformed['image']
    transformed_image = torch.from_numpy(transformed_image).permute(2, 0, 1).unsqueeze(0)

    pred = model(transformed_image.to(device)).cpu().detach()
    pred = pred.permute(0, 2, 3, 1)
    pred = pred.numpy()
    pred = np.concatenate(pred, axis=1)
    pred = np.squeeze(pred)
    return pred

inputs = gr.inputs.Image(type="filepath", label="Upload your picture here.")
outputs = gr.outputs.Image(type="numpy", label="Prediction will appear here")

app = gr.Interface(fn=predict, inputs=inputs, outputs=outputs, title="Satellite Images Segmentation App")
app.launch()