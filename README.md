# Paprica_detector
Paprica instance segmentation  using  Mask R-CNN. 


## Requirements

* Tensorflow 
* open-cv python

Install all the python dependencies using pip:
`pip install -r requirements.txt`

## Run

# Detection 
```
python3 paprika_detector.py

Infer Mask R-CNN to detect paprika.

optional arguments:
  -h, --help            show this help message and exit
  -ip IMAGES_PATH, --images_path IMAGES_PATH
                        Directory containing images to be inferred
  -sp SAVE_PATH, --save_path SAVE_PATH
                        Path for storing the prediction
  -wp WEIGHTS_PATH, --weights_path WEIGHTS_PATH
                        Path to weights .h5 file of mask R-CNN model
  --video               Predict from video camera
  --gpu                 whether to use gpu
  ```

