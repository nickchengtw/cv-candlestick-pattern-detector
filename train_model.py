# import the necessary packages
import argparse
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import lib.dataset as dataset
from lib.conf import Conf

# construct the argument parser and parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to the configuration file")
ap.add_argument(
    "-n",
    "--hard-negatives",
    type=int,
    default=-1,
    help="flag indicating whether or not hard negatives should be used",
)
args = vars(ap.parse_args())

# load the configuration file and the initial dataset
print("[INFO] loading dataset...")
conf = Conf(args["conf"])
(data, labels) = dataset.load_dataset(conf["features_path"], "features")

# check to see if the hard negatives flag was supplied
if args["hard_negatives"] > 0:
    print("[INFO] loading hard negatives...")
    (hardData, hardLabels) = dataset.load_dataset(conf["features_path"], "hard_negatives")
    data = np.vstack([data, hardData])
    labels = np.hstack([labels, hardLabels])

# train the classifier
print("[INFO] training classifier...")
model = SVC(kernel="linear", C=conf["C"], probability=True, random_state=42)
# model = RandomForestClassifier(n_estimators=conf['n_estimators'], random_state=42)
model.fit(data, labels)

# dump the classifier to file
print("[INFO] dumping classifier...")
with open(conf["classifier_path"], "wb") as f:
    f.write(pickle.dumps(model))
