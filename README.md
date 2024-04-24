## U-Net for Water Segmentation

In this project, the primary objective was to develop an AI model capable of accurately segmenting water bodies from satellite images.

### How to Train

1. Clone this repository:
    
    `git clone https://github.com/transiteration/unet-water-segmentation.git` 
    
2. To train the model on GPU locally, CUDA toolkit must be installed. The resulted model was trained on:
    - CUDA 11.8
    - Python 3.9
    - PyTorch 2.1.2
3. Install all python dependicies:
    
    `pip install -r requirements.txt`
    
4. Download and extract dataset from:
    
    https://drive.google.com/file/d/1ycnrrZOhYckGJgGeXSJvOI5s77PRcU5g/view?usp=drive_link
    
5. Run `experiment.py` with training parameter arguments, for example:
    
    `python3 experiment.py --dataset_loc path/to/dataset --num_workers 2 --num_epochs 10 --batch_size 32 --model_path path/to/save/model.pth`
    
6. Run `evaluate.py` on test set to evaluate model's performance, for example:

    `python3 evaluate.py --dataset_loc path/to/dataset --model_path path/to/saved/model.pth --num_workers 2 --batch_size 32`

7. Run `onnx_convert.py` to convert PyTorch model to ONNX runtime model, for example:

    `python3 onnx_convert.py --model_path path/to/saved/model.pth --onnx_path path/to/save/model.onnx`

7. Run `app.py` with path to ONNX model to inference custom pictures using gradio, for example:
    
    `python3 app.py --onnx_path path/to/saved/model.onnx`

#### Report

Read the full [notion report](https://www.notion.so/thankscarbon/U-Net-Model-for-Water-Segmentation-9bacf3912dc148098f4dd3b3473326d0) here.
