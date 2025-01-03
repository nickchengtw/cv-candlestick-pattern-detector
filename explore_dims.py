import xml.etree.ElementTree as ET
import numpy as np
import argparse

from utils.label import *
from lib.conf import Conf

# construct the argument parser and parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to the configuration file")
args = vars(ap.parse_args())

# load the configuration file
conf = Conf(args["conf"])

widths = []
heights = []

train_root = conf["image_dataset"]
batches = conf["image_batches"]
xml_file = "10_imglab.xml"
for image_file, boxes in load_label_batches(train_root, batches, xml_file):
    print(f"Image: {image_file}")
    for (top, left, width, height) in boxes:  # Find all <box> tags within the image
        # Calculate size and aspect ratio
        size = width * height
        aspect_ratio = width / height

        widths.append(width)
        heights.append(height)
        print(f"    Size: {size}, Aspect Ratio: {aspect_ratio:.2f}")

# compute the average of both the width and height lists
(avgWidth, avgHeight) = (np.mean(widths), np.mean(heights))
print("[INFO] avg. width: {:.2f}".format(avgWidth))
print("[INFO] avg. height: {:.2f}".format(avgHeight))
print("[INFO] aspect ratio: {:.2f}".format(avgWidth / avgHeight))