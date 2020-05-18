import argparse
import os

import numpy as np
import paprika
import src.mrcnn.model as modellib
import tensorflow as tf
from src.mrcnn import utils
from src.mrcnn.config import Config
from tqdm import tqdm

# Root directory of the project
ROOT_DIR = os.path.abspath("./")


class InferenceConfig(Config):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.9
    NAME = "paprika"
    NUM_CLASSES = 1 + 1  # Background + paprika


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


# Compute VOC-style Average Precision
def compute_batch_ap(image_ids):
    """

    Args:
        image_ids: list of image_ids

    Returns:
        APs: Average Precisions

    """

    APs = []
    for image_id in tqdm(image_ids):
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset, config,
                                   image_id, use_mini_mask=False)
        # Run object detection
        results = model.detect([image], verbose=0)
        # Compute AP
        r = results[0]
        AP, precisions, recalls, overlaps = \
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r['rois'], r['class_ids'], r['scores'], r['masks'])
        APs.append(AP)
    return APs


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Evaluate Mask R-CNN mAP')

    parser.add_argument('-dp', '--data_path', default='data/input/',
                        type=str, help='Directory containing images to be inferred')

    parser.add_argument('-wp', '--weights_path',
                        default='data/mask_rcnn_paprika_final.h5',
                        type=str,
                        help="Path to weights .h5 file of mask R-CNN model")

    parser.add_argument('-e', '--evaluate', default='val',
                        type=str, help='val/test for evaluation')

    parser.add_argument('--gpu', help='whether to use gpu',
                        dest='gpu',
                        action='store_true')

    args = parser.parse_args()

config = paprika.PaprikaConfig()
PAPRIKA_DIR = os.path.join(ROOT_DIR, args.data_path)
config = InferenceConfig()
config.display()

DEVICE = "/cpu:0"
if args.gpu:
    DEVICE = "/gpu:0"
    print("Prediction using GPU")
TEST_MODE = "inference"

# Load validation dataset
dataset = paprika.PaprikaDataset()
dataset.load_paprika(PAPRIKA_DIR, args.evaluate)

print("Evaluation on ", args.evaluate)

# Must call before using the dataset
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference",
                              model_dir=os.path.join(ROOT_DIR, "logs"),
                              config=config)

# Load weights
print("Loading weights ", args.weights_path)
model.load_weights(args.weights_path, by_name=True)

print("Evaluating......")
APs = compute_batch_ap(dataset.image_ids)
print("mAP @ IoU=50: ", np.mean(APs))
