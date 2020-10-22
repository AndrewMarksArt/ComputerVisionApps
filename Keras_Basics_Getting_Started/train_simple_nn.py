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

# split the data into training and testing sets, 75/25 split
(X_train, X_test, y_train, y_test) = train_test_split(data, labels,
                                                      test_size=0.25,
                                                      random_state=42)

# convert the labels from integers to vectors
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

# very simple model -> input layer, 2 hidden layers, output layer
model = Sequential()
model.add(Dense(1024, input_shape=(3072, ), activation="sigmoid"))
model.add(Dense(512, activation="sigmoid"))
model.add(Dense(len(lb.classes_), activation="softmax"))

# compile the model
# initialize learning rate and # of epochs
INIT_LR = 0.01
EPOCHS = 80

# use SGD as our optimizer and categorical cross-entropy loss
print("[INFO] training network...")
opt = SGD(lr=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train the model
H = model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test),
              epochs=EPOCHS, batch_size=32)

# model evaluation
print("[INFO] evaluating the model...")
predictions = model.predict(x=X_test, batch_size=32)
print(classification_report(y_test.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=lb.classes_))

# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="training_loss")
plt.plot(N, H.history["val_loss"], label="validation_loss")
plt.plot(N, H.history["accuracy"], label="accuracy")
plt.plot(N, H.history["val_accuracy"], label="validation_accuracy")
plt.title("Training Loss and Accuracy for Simple CNN")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])

# save the model and label binarizer to disk
print("[INFO] serializing network and label binarizer...")
model.save(args["model"], save_format="h5")
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()

