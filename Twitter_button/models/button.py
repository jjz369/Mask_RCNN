#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Mask R-CNN

Configurations and data loading for the twitter following button detection

This code is modified based on the Shape datasets sample configuration and 
Balloon
Copyright (c) 2017 Matterport, Inc.

Training on Twitter following button detection by Jingjie Zhang (jjzhang369@gmail.com)

Created on Wed Feb  5 09:16:06 2020

"""

import os
import sys
import json
import random
import numpy as np
import skimage.draw
import cv2
import matplotlib
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)

CURR_DIR = os.path.abspath("./")

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# Path to pre-trained weights  
MODEL_DIR = os.path.join(CURR_DIR, "button_logs")
COCO_MODEL_PATH = os.path.join(CURR_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

############################################################
#   Configuration
############################################################
class ButtonConfig(Config):
    """Configuration for training on the Twitter following page screenshot 
    figures. Derives from the base Config class and overrides values specific
    to the twitter following button dataset.
    """
    # Give the configuration a recognizable name
    NAME = "button"

    # Train on 2 images per GPU. We can put multiple images on each
    # GPU because the images are small. 
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 2  # background + 1 button class

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # Use a smaller anchor because some of images and objects are small 
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
    

############################################################
#   Datasets
############################################################
class ButtonDataset(utils.Dataset):
    """
    load pre-treated twitter button images and masks from annotations
        
    """

    def load_button(self, dataset_dir, subset):
        """
        Load a subset of the twitter following button image datasets.

        Parameters
        ----------
        dataset_dir : Root directory of the dataset
        subset : subsets to load: train or val

        """
        
        # Add classes. We have only one class to add.
        self.add_class("button", 1, "button")

        # Train or validation datasets?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Add images
        annotations = json.load(open(os.path.join(dataset_dir, "via_export_json.json")))
        annotations = list(annotations.values())
        annotations = [a for a in annotations if a['regions']]


        # Get the x, y coordinaets of points of the rectangles that make up
        # the outline of each object instance. These are stores in the
        # shape_attributes (see json format above)
        # The if condition is needed to support VIA versions 1.x and 2.x.
        for a in annotations:
            if type(a['regions']) is dict:
                rectangles = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                rectangles = [r['shape_attributes'] for r in a['regions']] 

            # load_mask() needs the image size to convert rectangles to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "button",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                rectangles=rectangles)

    def load_mask(self, image_id):
        """
        Generate instance masks for an image. 
        This function is modified because instead of using polygons, I used a
        rectangle for the annotations.

        Parameters
        ----------
        image_id : the loaded internel image_id 

        Returns
        -------
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.

        """
        # If not a button dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "button":
            return super(self.__class__, self).load_mask(image_id)

        # Convert rectangles to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["rectangles"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["rectangles"]):
          start = (p['y'], p['x'])
          extent = (p['y']+p['height'], p['x']+p['width'])
          # Get indexes of pixels inside the rectangle and set them to 1
          rr, cc = skimage.draw.rectangle(start, extent)
          mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "button":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
    
def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = ButtonDataset()
    dataset_train.load_button(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ButtonDataset()
    dataset_val.load_button(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE, 
                epochs=1, 
                layers='heads')
    
    print("Training network all")
    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE / 10,
                epochs=2, 
                layers="all")
    
def detect(model, image_path = None):
    assert image_path, "Argument --image_path is required for detection"
    

    # Read image
    image = skimage.io.imread(args.image)
    # Detect objects
    r = model.detect([image], verbose=1)[0]
    
    print("The bounding boxes are: " r[rois])
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])
    
    
    
if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect following buttons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/button/dataset/",
                        help='Directory of the Button dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file ")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply detect')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.image, "Provide --image to detect"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = ButtonConfig()
    else:
        class InferenceConfig(ButtonConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "detect":
        detect(model)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))    