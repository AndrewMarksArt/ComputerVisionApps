# import packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

# construct argument parse
ap = argparse.ArgumentParser()

# parse the arguments
# path to video (optional) if provided use video file
# if not, access the webcam
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")

# --buffer is the optional max size of our deque which
# maintains a list of the previous (x,y) coordinates of
# what we are tracking
ap.add_argument("-b", "--buffer", type=int, default=64,
    help="max buffer size")

# parse command line arguments
args = vars(ap.parse_args())

# define the lower and upper boundaries of the object
# in the HSV color space, then initialize the list of
# tracking points

# using color 'green'
green_lower = (29, 86, 6)
green_upper = (64, 255, 255)
points = deque(maxlen=args["buffer"])

# if the video path was not supplied, get reference
# to the webcam
if not args.get("video", False):
    video_stream = VideoStream(src=0).start()

# if video path provided, grab reference to file
else:
    video_stream = cv2.VideoCapture(args["video"])

# allow the camera or video file to load
time.sleep(2.0)

# loop over video or camera frames
while True:
    # grabe the current frame
    frame = video_stream.read()

    # handle the frame from VideoCapture or VideoStream
    frame = frame[1] if args.get("video", False) else frame

    # if viewing video and we did not grabe a frame,
    # then we are at the end of the video
    if frame is None:
        break

    # resize, blur, and convert frame to HSV color space
    frame = imutils.resize(frame, width=1200)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # construct a mask for the object, then perform a series
    # of dilations and erosions to remove any small blobs
    # left in the mask
    mask = cv2.inRange(hsv, green_lower, green_upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current
    # center of the object
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    center = None

    # only proceed if at least 1 contour was found
    if len(contours) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(contours, key=cv2.contourArea)
        ((x,y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"]/M["m00"]), int(M["m01"]/int(M["m00"])))

        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and cnetroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    # update the points queue
    points.appendleft(center)

    # loop over set of tracked points
    for i in range(1, len(points)):
        # if either of the tracked points are None, ignore them
        if points[i -1] is None or points[i] is None:
            continue

        # otherwise, compute the thickness of the line and draw
        # and draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i +1)) * 2.5)
        cv2.line(frame, points[i - 1], points[i], (0, 0, 255), thickness)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    key =cv2.waitKey(1) & 0xFF

    # if the 'q' jey is pressed, stop the loop
    if key == ord("q"):
        break

# if no video file stop the camera
if not args.get("video", False):
    video_stream.stop()

# otherwise, release the camera
else:
    video_stream.release()

# close the window
cv2.destroyAllWindows()
