# Paprica_detector
Paprica instance segmentation  using  Mask R-CNN. 


## Requirements

* Tensorflow 
* open-cv python

Install all the python dependencies using pip:

`pip install -r requirements.txt`


## Train the Paprika model

Train a new model starting from pre-trained COCO weights
```
python3 paprika.py train --dataset=/path/to/balloon/dataset --weights=coco
```

Resume training a model that you had trained earlier
```
python3 paprika.py train --dataset=/path/to/balloon/dataset --weights=last
```

Train a new model starting from ImageNet weights
```
python3 paprika.py train --dataset=/path/to/balloon/dataset --weights=imagenet
```

The code in `paprika.py` is set to train for 3K steps (30 epochs of 100 steps each), and using a batch size of 2. 
Update the schedule to fit your needs.


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

