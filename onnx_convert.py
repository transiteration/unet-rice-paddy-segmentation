import onnx
import torch
import argparse
import onnxruntime
import numpy as np
from model import UNet

def pth_to_onnx(model_path: str = None,
                onnx_path: str = None) -> None:
    model = UNet(1, 1)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    dummy_input = torch.randn(1, 1, 256, 256)
    torch_out = model(dummy_input)

    torch.onnx.export(model,
                    dummy_input,
                    onnx_path,
                    verbose=True)

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(onnx_path)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=None, help="Path to saved PyTorch model.")
    parser.add_argument("--onnx_path", default=None, help="Path to where save onnx model.")
    args = parser.parse_args()

    pth_to_onnx(model_path=args.model_path,
                onnx_path=args.onnx_path)




