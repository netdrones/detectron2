import os
import cv2
import argparse
import detectron2
import numpy as np
import os, json, cv2, random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

LOWER_RED = np.array([0,70,70])
UPPER_RED = np.array([20,200,150])

def rust_detect(path):

	img = cv2.imread(path)
	img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	# Range for lower red
	lower_red = np.array([0,70,70])
	upper_red = np.array([20,200,150])
	mask_lower = cv2.inRange(img_hsv, lower_red, upper_red)

	# Range for upper red
	lower_red = np.array([170,70,70])
	upper_red = np.array([180,200,150])
	mask_upper = cv2.inRange(img_hsv, lower_red, upper_red)

    mask = mask_lower + mask_upper

    return cv2.bitwise_and(img, img, mask=mask)

def main(args):

	# Get images
    images = []
    img_paths = os.listdir(args.img_dir)
    for img in img_paths:
        images.append(cv2.imread(os.path.join(args.img_dir, img)))

	# Get configs
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    for image in images:
        outputs = predictor(image)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Headless instance segmentation')
    parser.add_argument('--img_dir', help='path to images')
    args = parser.parse_args()

    main(args)
