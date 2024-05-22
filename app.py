import time
import argparse
import numpy as np
import onnxruntime
import gradio as gr
from PIL import Image

def predict(image_path: str = None) -> Image.Image:
    start_time = time.time()
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    image = image.resize((256, 256))
    image = np.array(image, dtype=np.float32) / 255.0
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)

    ort_inputs = {ort_session.get_inputs()[0].name: image}
    output = ort_session.run(None, ort_inputs)[0][0][0]

    output = (output - np.min(output)) / (np.max(output) - np.min(output))
    output = np.clip(output * 2 - 1, 0, 1) * 255
    output_image = Image.fromarray(output.astype(np.uint8))
    output_image = output_image.resize(original_size)
    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    return output_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_path", default=None, help="Path to saved onnx model.")
    args = parser.parse_args()

    ort_session = onnxruntime.InferenceSession(args.onnx_path)

    inputs = gr.inputs.Image(type="filepath", label="Upload your picture here.")
    outputs = gr.outputs.Image(type="pil", label="Prediction will appear here")
    app = gr.Interface(fn=predict, inputs=inputs, outputs=outputs, title="Satellite Images Segmentation App")
    app.launch(share=False)