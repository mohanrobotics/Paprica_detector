import paprika
import os
import src.mrcnn.model as modellib
import numpy as np
import tensorflow as tf
from src.mrcnn import utils

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

config = paprika.PaprikaConfig()
PAPRIKA_DIR = os.path.join(ROOT_DIR, "data/input/")

class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


# Compute VOC-style Average Precision
def compute_batch_ap(image_ids):
    APs = []
    for image_id in image_ids:
	print('Predicting ',image_id)
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, config,
                                   image_id, use_mini_mask=False)
        # Run object detection
        results = model.detect([image], verbose=0)
        # Compute AP
        r = results[0]
        AP, precisions, recalls, overlaps =\
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                              r['rois'], r['class_ids'], r['scores'], r['masks'])
        APs.append(AP)
    return APs


config = InferenceConfig()
config.display()

DEVICE = "/cpu:0"
TEST_MODE = "inference"


# Load validation dataset
dataset = paprika.PaprikaDataset()
dataset.load_paprika(PAPRIKA_DIR, "val")


# Must call before using the dataset
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)


weights_path = "/gluster/scratch/mmuthuraja/paprika_detection/Mask_RCNN/paprika_logs/mask_rcnn_paprika_final.h5"

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)


APs = compute_batch_ap(dataset.image_ids)
print("mAP @ IoU=50: ", np.mean(APs))
