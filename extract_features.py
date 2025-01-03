# import the necessary packages
from sklearn.feature_extraction.image import extract_patches_2d
from imutils import paths
import numpy as np
import progressbar
import argparse
import random
import cv2

import lib.helpers as helpers
from lib.hog import HOG
import lib.dataset as dataset
from lib.conf import Conf
from utils.augmentation import *
from utils.preprocess import process_image
from utils.label import *

# construct the argument parser and parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to the configuration file")
args = vars(ap.parse_args())

# load the configuration file
conf = Conf(args["conf"])

# initialize the HOG descriptor along with the list of data and labels
hog = HOG(
    orientations=conf["orientations"],
    pixelsPerCell=tuple(conf["pixels_per_cell"]),
    cellsPerBlock=tuple(conf["cells_per_block"]),
    normalize=conf["normalize"],
)
data = []
labels = []

print("[INFO] describing training ROIs...")

train_root = conf["image_dataset"]
batches = conf["image_batches"]
xml_file = "10_imglab.xml"
for image_file, boxes in load_label_batches(train_root, batches, xml_file):
    print(f"Image: {image_file}")
    image = cv2.imread(image_file)
    image = process_image(image)
    
    for (top, left, width, height) in boxes:  # Find all <box> tags within the image
        bb = (top, top+height, left, left+width)
        print(bb)

        roi = helpers.crop_ct101_bb(image, bb, padding=conf["offset"], dstSize=tuple(conf["window_dim"]))

        """
        flip peak we get valley
        """
        if conf["vertical_flip"]:
            roi = cv2.flip(roi, 0)

        # define the list of ROIs that will be described, based on whether or not the
        # horizontal flip of the image should be used
        rois = (roi, cv2.flip(roi, 1)) if conf["use_flip"] else (roi,)

        rois = (*rois,
                *scale_augmentation(rois[0], 1.1, conf["window_dim"]),
                *scale_augmentation(rois[1], 1.1, conf["window_dim"]),
                *scale_augmentation(rois[0], 1.2, conf["window_dim"]),
                *scale_augmentation(rois[1], 1.2, conf["window_dim"])
                )
        
        # loop over the ROIs
        for roi in rois:
            # cv2.imshow("image", roi)
            cv2.waitKey(0)
            
            # extract features from the ROI and update the list of features and labels
            features = hog.describe(roi)
            data.append(features)
            labels.append(1)

neg_root = conf["image_distractions"]
dstPaths = list(paths.list_images(neg_root))
print("[INFO] describing distraction ROIs...")
for i in np.arange(0, conf["num_distraction_images"]):
    image = cv2.imread(random.choice(dstPaths))
    image = process_image(image)
    patches = extract_patches_2d(image, tuple(conf["window_dim"]), max_patches=conf["num_distractions_per_image"])

    # loop over the patches
    for patch in patches:
        # extract features from the patch, then update the data and label list
        features = hog.describe(patch)
        data.append(features)
        labels.append(-1)

# dump the dataset to file
print("[INFO] dumping features and labels to file...")
dataset.dump_dataset(data, labels, conf["features_path"], "features")
