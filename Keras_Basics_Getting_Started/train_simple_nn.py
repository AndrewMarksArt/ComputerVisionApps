# set matplotlib so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from imutils import paths

import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset of images")
ap.add_argument("-m", "--model", required=True,
                help="path to output trained model")
ap.add_argument("-l", "--label-bin", required=True,
                help="path to output label binarizer")
ap.add_argument("-p", "--plot", required=True,
                help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize data and labels
print("[INFO] loading images...")
data = []
labels = []

# get image path and randomly shuffle the images
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
    # load the image, resize to 32 x 32 ignoring aspect ratio
    # flatten the image into 32 x 32 x 3 vector and add the vector
    # to the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (32, 32)).flatten()
    data.append(image)

    # get class label from image path, update labels list
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# scale pixel intensities from 0-255 to 0-1
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
