## Scaling-ML

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
    
    https://drive.google.com/file/d/1C7xLuQTF3jbbIqrgOQFOnxHnnaG5soD4/view?usp=drive_link
    
5. Run `experiment.py` **with training parameter arguments, for exampe:
    
    `python3 experiment.py --dataset_loc path/to/dataset --num_workers 2 --num_epochs 10 --batch_size 32 --model_path path/to/save/model`
    
6. Run `app.py` with path to trained model to inference custom pictures:
    
    `python3 app.py --model_path path/to/saved/model`
