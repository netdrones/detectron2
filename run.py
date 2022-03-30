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
        outputs = predictor(im)
        print(outputs["instances"].pred_classes)
        print(outputs["instances"].pred_boxes)
        breakpoint()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Headless instance segmentation')
    parser.add_argument('--img_dir', help='path to images')
    args = parser.parse_args()

    main(args)
