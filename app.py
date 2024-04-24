import argparse
import numpy as np
import onnxruntime
import gradio as gr
from PIL import Image

def predict(image_path):
    image = Image.open(image_path).convert("L")
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(np.expand_dims(image, axis=0), axis=0)
    ort_inputs = {ort_session.get_inputs()[0].name: image}
    ort_outputs = ort_session.run(None, ort_inputs)
    output = ort_outputs[0][0][0]
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_path", default=None, help="Path to saved onnx model.")
    args = parser.parse_args()

    ort_session = onnxruntime.InferenceSession(args.onnx_path)

    inputs = gr.inputs.Image(type="filepath", label="Upload your picture here.")
    outputs = gr.outputs.Image(type="numpy", label="Prediction will appear here")
    app = gr.Interface(fn=predict, inputs=inputs, outputs=outputs, title="Satellite Images Segmentation App")
    app.launch(share=True)