import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob 
import skimage
import src.mrcnn.model as modellib
import tensorflow as tf
from src.mrcnn.config import Config
from src.mrcnn.visualize import display_instances

from src.mrcnn.visualize import visualize_boxes_and_labels_on_image_array

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ROOT_DIR = os.path.abspath("./data/")


class InferenceConfig(Config):
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	DETECTION_MIN_CONFIDENCE = 0.9
	NAME = "paprika"
	NUM_CLASSES = 1 + 1  # Background + paprika

def detect_and_display(model, images_path=None, save_path=None ):



	images_path_list = glob.glob(os.path.join(images_path, '*'))
	print(images_path_list)
	for idx, image_path in enumerate(images_path_list):
		if image_path.endswith('jpg' or 'png' or 'jpeg'):

			# Run model detection and generate the color splash effect
			print("Running on {}".format(image_path.split('/')[-1]))
			# Read image
			image = skimage.io.imread(image_path)
			# Detect objects
			results = model.detect([image], verbose=1)[0]

			save_dir = None
			if save_path:
				save_dir = os.path.join(save_path,'prediction_{}.png'.format(idx))

			visualize_boxes_and_labels_on_image_array(image,
                                      results['rois'],
                                      results['class_ids'],
                                      results['scores'],
                                      {0:{'name':'BG'}, 1:{'name':'paprika'}},
                                      instance_masks=results['masks'],
                                      keypoints=None,
                                      use_normalized_coordinates=False,
                                      max_boxes_to_draw=20,
                                      min_score_thresh=.5,
                                      agnostic_mode=False,
                                      line_thickness=4)

			cv2.imwrite(save_dir, cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR))




def display_video(model, camera_id = 0):


	cap = cv2.VideoCapture(camera_id)

	count = 0 ; 

	while(True):
		# Capture frame-by-frame
		ret, frame = cap.read()

		image = frame[..., ::-1]
		scale_percent = 60 # percent of original size
		width = int(image.shape[1] * scale_percent / 100)
		height = int(image.shape[0] * scale_percent / 100)
		dim = (width, height)
		# resize image
		image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
		results = model.detect([image], verbose=1)[0]

		visualize_boxes_and_labels_on_image_array(image,
                                          results['rois'],
                                          results['class_ids'],
                                          results['scores'],
                                          {0:{'name':'BG'}, 1:{'name':'paprika'}},
                                          instance_masks=results['masks'],
                                          keypoints=None,
                                          use_normalized_coordinates=False,
                                          max_boxes_to_draw=20,
                                          min_score_thresh=.5,
                                          agnostic_mode=False,
                                          line_thickness=4)



		# Display the resulting frame
		cv2.imshow('frame',cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break


		# if key == 27:#if ESC is pressed, exit loop
	 #        cv2.destroyAllWindows()
	 #        break

		count+=1 

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()




if __name__ == '__main__':

	# Parse command line arguments
	parser = argparse.ArgumentParser(
		description='Infer Mask R-CNN to detect paprika.')

	parser.add_argument('-ip', '--images_path', default='data/input/',
					type=str, help='Directory containing images to be inferred')

	parser.add_argument('-sp', '--save_path', default='data/output/',
					type=str, help='Path for storing the prediction')

	parser.add_argument('-wp', '--weights_path', default='data/mask_rcnn_paprika_latest.h5',
					type=str, help='Path to weights .h5 file of mask R-CNN model')

	parser.add_argument('--video', dest='video',
						help='Predict from video camera',
						action='store_true')
	parser.add_argument('--gpu', dest='gpu',
						help='whether to use gpu',
						action='store_true')

	args = parser.parse_args()

DEVICE = "/cpu:0"
if args.gpu:
	DEVICE = "/gpu:0"
	print("Prediction using GPU")

# Create model in inference mode
config = InferenceConfig()
with tf.device(DEVICE):
	model = modellib.MaskRCNN(mode="inference", model_dir=ROOT_DIR,
							  config=config)

# Load weights
print("Loading weights ", args.weights_path)
model.load_weights(args.weights_path, by_name=True)

if args.video:
	display_video(model, camera_id = 0)
else:
	detect_and_display(model, images_path=args.images_path, save_path = args.save_path)


