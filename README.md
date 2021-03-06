# Paprica_detector
Paprica instance segmentation  using  Mask R-CNN.
Most of core algorithm code was based on [Mask R-CNN implementation by Matterport, Inc. ](https://github.com/matterport/Mask_RCNN)

![Detections](/extras/detections.gif)


## Requirements

* Tensorflow 
* open-cv python

Install all the python dependencies using pip:

`pip install -r requirements.txt`

## Run Jupyter notebooks
Open the `inspect_paprika_data.ipynb` or `inspect_paprika_model.ipynb` Jupter notebooks. You can use these notebooks to explore the dataset and run through the detection pipelie step by step.

## Project notebooks
```
├── data
│   ├── input
│   │   ├── README.md
│   │   ├── train
│   │   │   ├── <train images>
│   │   │   ├── via_region_data.json
│   │   └── val
│   │       ├── <val images >
│   │   │   ├── via_region_data.json
│   │   ├── test
│   │   │   ├── <test images>
│   ├── mask_rcnn_paprika_final.h5
│   └── output
│       ├── README.md
│       ├── test_op
│       │   ├── <test images predicted saved here>
│       └── val_op
│           ├── <Val images predicted saved here>
└── src
|   └── mrcnn
│       ├──config.py
│       ├── __init__.py
│       ├── model.py
│       ├── utils.py
│       ├── visualize.py
├── paprika.py
├── paprika_detector.py
├── paprika_evaluator.py
├── inspect_paprika_data.ipynb
├── inspect_paprika_model.ipynb
├── requirements.txt
├── extras
│   └── detections.gif
├── README.md
```

## Train the Paprika model

Train a new model starting from pre-trained COCO weights
```
python3 paprika.py train --dataset=data/input/ --weights=coco
```

Resume training a model that you had trained earlier
```
python3 paprika.py train --dataset=data/input/ --weights=last
```

Train a new model starting from ImageNet weights
```
python3 paprika.py train --dataset=data/input/ --weights=imagenet
```

The code in `paprika.py` is set to train for 3K steps (30 epochs of 100 steps each), and using a batch size of 2. 
Update the schedule to fit your needs.


## Detection using trained model
```
python3 paprika_detector.py -ip=data/input/val -sp=data/input/val_op -wp=data/mask_rcnn_paprika_final.h5

usage: paprika_detector.py [-h] [-ip IMAGES_PATH] [-sp SAVE_PATH]
                           [-wp WEIGHTS_PATH] [--video] [--gpu]
                           [--display_pred]

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
  --display_pred        whether to display the prediction
  ```
  
  
## Evaluating using trained model
```
python paprika_evaluator.py -sp=data/input/ -e=val -wp=data/mask_rcnn_paprika_final.h5
usage: paprika_evaluator.py [-h] [-dp DATA_PATH] [-wp WEIGHTS_PATH]
                            [-e EVALUATE] [--gpu]

Evaluate Mask R-CNN mAP

optional arguments:
  -h, --help            show this help message and exit
  -dp DATA_PATH, --data_path DATA_PATH
                        Directory containing images to be inferred
  -wp WEIGHTS_PATH, --weights_path WEIGHTS_PATH
                        Path to weights .h5 file of mask R-CNN model
  -e EVALUATE, --evaluate EVALUATE
                        val/test for evaluation
  --gpu                 whether to use gpu
  ```
