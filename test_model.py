# import the necessary packages
import numpy as np
import imutils
import argparse
import pickle
import cv2

from lib.hog import HOG
from lib.conf import Conf
from lib.nms import non_max_suppression
from lib.object_detection import ObjectDetector
from utils.preprocess import process_image
from utils.pattern import *


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-c", "--conf", required=True, help="path to the configuration file")
ap.add_argument("-i", "--image", required=True, help="path to the image to be classified")
args = vars(ap.parse_args())

# load the image and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=min(640, image.shape[1]))
processed_image = process_image(image)

marked = image.copy()
peaks = []
valleys = []

def test_image_peak(conf_path):
    conf = Conf(conf_path)
    # load the classifier, then initialize the Histogram of Oriented Gradients descriptor
    # and the object detector
    model = pickle.loads(open(conf["classifier_path"], "rb").read())
    hog = HOG(orientations=conf["orientations"],
            pixelsPerCell=tuple(conf["pixels_per_cell"]),
            cellsPerBlock=tuple(conf["cells_per_block"]),
            normalize=conf["normalize"], block_norm="L1")
    od = ObjectDetector(model, hog)

    # detect objects in the image and apply non-maxima suppression to the bounding boxes
    (boxes, probs) = od.detect(processed_image, conf["window_dim"],
                            minSize=conf["pyramid_min_size"],
                            winStep=conf["window_step"],
                            pyramidScale=conf["pyramid_scale"],
                            minProb=conf["min_probability"])
    pick = non_max_suppression(np.array(boxes), probs, conf["overlap_thresh"])
    
    image_copy = image.copy()
    orig = image.copy()
    # loop over the original bounding boxes and draw them
    for (startX, startY, endX, endY) in boxes:
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 2)
    # loop over the allowed bounding boxes and draw them
    for (startX, startY, endX, endY) in pick:
        cv2.rectangle(image_copy, (startX, startY), (endX, endY), (0, 255, 0), 2)

    for (startX, startY, endX, endY) in pick:
        top_center = ((startX + endX) // 2, startY)
        peaks.append(top_center)
        cv2.circle(marked, top_center, radius=5, color=(0, 255, 0), thickness=-1)
        
    # show the output images
    cv2.imshow("Peak Original", orig)
    cv2.imshow("Peak Image", image_copy)
    

def test_image_valley(conf_path):
    conf = Conf(conf_path)
    # load the classifier, then initialize the Histogram of Oriented Gradients descriptor
    # and the object detector
    model = pickle.loads(open(conf["classifier_path"], "rb").read())
    hog = HOG(orientations=conf["orientations"],
            pixelsPerCell=tuple(conf["pixels_per_cell"]),
            cellsPerBlock=tuple(conf["cells_per_block"]),
            normalize=conf["normalize"], block_norm="L1")
    od = ObjectDetector(model, hog)

    # detect objects in the image and apply non-maxima suppression to the bounding boxes
    (boxes, probs) = od.detect(processed_image, conf["window_dim"],
                            minSize=conf["pyramid_min_size"],
                            winStep=conf["window_step"],
                            pyramidScale=conf["pyramid_scale"],
                            minProb=conf["min_probability"])
    pick = non_max_suppression(np.array(boxes), probs, conf["overlap_thresh"])
    
    image_copy = image.copy()
    orig = image.copy()
    # loop over the original bounding boxes and draw them
    for (startX, startY, endX, endY) in boxes:
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 2)
    # loop over the allowed bounding boxes and draw them
    for (startX, startY, endX, endY) in pick:
        cv2.rectangle(image_copy, (startX, startY), (endX, endY), (0, 255, 0), 2)

    for (startX, startY, endX, endY) in pick:
        top_center = ((startX + endX) // 2, endY)
        valleys.append(top_center)
        cv2.circle(marked, top_center, radius=5, color=(0, 0, 255), thickness=-1)
        
    # show the output images
    cv2.imshow("Valley Original", orig)
    cv2.imshow("Valley Image", image_copy)

print("Finding all peaks")
test_image_peak('peak.json')
print("Finding all valleys")
test_image_valley('valley.json')

cv2.imshow("Marked", marked)
with open("coord.pickle", "wb") as file:
    pickle.dump([peaks, valleys], file)
    


patterns = find_valley_peak_pattern(peaks, valleys, peak_tolerance=50)
print(patterns)
for pattern in patterns:
    image = plot_pattern(image, pattern)
cv2.imshow("Image with Patterns", image)


cv2.waitKey(0)
