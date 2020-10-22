"""
use the saved model to predict on new images
"""

# imports
from tensorflow.keras.models import load_model
import argparse
import pickle
import cv2

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-m", "--model", required=True,
                help="path to trained keras model")
ap.add_argument("-l", "--labels", required=True,
                help="path to label binarizer")
ap.add_argument("-w", "--width", type=int, default=28,
                help="target spatial dimension width")
ap.add_argument("-h", "--height", type=int, default=28,
                help="target spatial dimension height")
ap.add_argument("-f", "--flatten", type=int, default=-1,
                help="should we flatten the image")
args = vars(ap.parse_args())

# load the image and resize it to the target spatial dimensions
image = cv2.imread(args["image"])
output = image.copy()
image = cv2.resize(image, (args["width"], args["height"]))

# scale the pixel values to between [0, 1]
image = image.astype("float") / 255.0

# check to see if the image needs to be flattened
if args["flatten"] > 0:
    image = image.flatten()
    image = image.reshape((1, image.shape[0]))
# if we dont need to flatten, add the batch dimensions
else:
    image = image.reshape((1, image.shape[0], image.shape[1],
                           image.shape[2]))

# load the model and the label binarizer
print("[INFO] loading network and label binarizer...")
model = load_model(args["model"])
lb = pickle.load(open(args["labels"], "rb").read())


