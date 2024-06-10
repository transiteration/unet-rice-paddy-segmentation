import time
import argparse
import numpy as np
import onnxruntime
import gradio as gr
from PIL import Image

def predict(image_path: str = None) -> tuple:
    start_time = time.time()
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    image = image.resize((512, 512))
    image = np.array(image, dtype=np.float32) / 255.0
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)

    ort_inputs = {ort_session.get_inputs()[0].name: image}
    output = ort_session.run(None, ort_inputs)[0][0][0]

    min_max_image = (output - np.min(output)) / (np.max(output) - np.min(output))
    min_max_image = np.clip(min_max_image * 2 - 1, 0, 1) * 255
    min_max_image = Image.fromarray(min_max_image.astype(np.uint8))
    min_max_image = min_max_image.resize(original_size)

    binary_image = np.where(output >= 0.5, 1, 0) * 255
    binary_image = Image.fromarray(binary_image.astype(np.uint8))
    binary_image = binary_image.resize(original_size)

    end_time = time.time()
    processing_time = f"{end_time - start_time:.2f} seconds"
    print(processing_time)
    return processing_time, min_max_image, binary_image 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_path", default=None, help="Path to saved onnx model.")
    args = parser.parse_args()

    ort_session = onnxruntime.InferenceSession(args.onnx_path)

    inputs = gr.Image(type="filepath", label="Upload your picture here.")
    outputs = [gr.Textbox(label="Processing time"),
               gr.Image(type="pil", label="MinMax Image"),
               gr.Image(type="pil", label="Binary Image")]

    app = gr.Interface(fn=predict,
                       inputs=inputs,
                       outputs=outputs,
                       allow_flagging="never",
                       title="Satellite Rice Paddy Images Segmentation App")
    app.launch(share=True)