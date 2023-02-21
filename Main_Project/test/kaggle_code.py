import cv2
import numpy as np
import os

import datetime
from skimage import io
import os
import random
import matplotlib.pyplot as plt
# matplotlib inline

import glob


def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)
    

# Starts capturing video
cap = cv2.VideoCapture(0)

fps = int(cap.get(cv2.CAP_PROP_FPS))

print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

while cap.isOpened():
    ret, frame = cap.read()
    th_f = segment(frame)
    cv2.imshow('Captured Frame', th_f)
    if cv2.waitKey(1) == ord('q'):
        break

    keypress = cv2.waitKey(1) & 0xFF



cap.release()
cv2.destroyAllWindows()