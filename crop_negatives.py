# import the necessary packages
from sklearn.feature_extraction.image import extract_patches_2d
from imutils import paths
import numpy as np
import progressbar
import argparse
import random
import cv2
import uuid

import lib.helpers as helpers
from lib.conf import Conf
from utils.augmentation import *
from utils.preprocess import process_image
from utils.label import load_labels

train_root = 'dataset/negitives_bb'
xml_file =  train_root + '/10_imglab_neg.xml'
for image_file, boxes in load_labels(train_root, xml_file):
    print(f"Image: {image_file}")
    image = cv2.imread(image_file)
    
    for (top, left, width, height) in boxes:  # Find all <box> tags within the image
        bb = (top, top+height, left, left+width)
        print(bb)

        roi = helpers.crop_ct101_bb(image, bb, padding=0, dstSize=(width, height))
        cv2.imwrite(f'dataset/negitives/{uuid.uuid4()}.png', roi)
