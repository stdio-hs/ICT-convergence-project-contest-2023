import pickle
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
#from typing import Self
labels = ['blank', 'fist', 'five', 'ok', 'thumbsdown', 'thumbsup']

def model_load(model_path):
    print('model_loading')
    #model = pickle.load(model_path + model_name)
    model = load_model(model_path)
    print(model.summary())
    return model

def _load_weights():
    try:
        model = model_load("hand_gesture_recog_model.h5")
        print(model.summary())
        # print(model.get_weights())
        # print(model.optimizer)
        return model
    except Exception as e:
        return None
    
def getPredict(model, frame):
    prediction = _predict(model, frame)
    if prediction is not None:
        #prediction = np.array(prediction)
        classFound = np.array(prediction)
        #classFound = np.asarray(prediction, dtype = int)
        label = _getLabel(classFound)
        return label 

    return prediction

def _predict(model, frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.resize(gray_image, (100, 120))
    gray_image = img_to_array(gray_image)
    gray_image = np.expand_dims(gray_image, axis = 0)

    prediction = model.predict(gray_image)
    return prediction

def _getLabel(predict):
    arg_class = np.argmax(predict)
    predict_label = labels[arg_class]
    return predict_label




#-------------------------------------------------------------
import cv2

#글로벌 변수 (배경)
bg = None


#cv2로 받은 비디오를 총괄하여 관리하는 클래스
#240 215
class VideoCustom():
    top, right, bottom, left = 30, 200, 130, 320
    #top, right, bottom, left = 30, 200, 270, 415

    def __init__(self, frame):
        self.customFrame(frame)
        self.cvGray(self.roi)
        

    def customFrame(self, frame):
        #프레임 크기 변형
        self.frame = cv2.resize(frame, (700, 700))
        #프레임 좌우 반전
        self.frame = cv2.flip(frame, 1)

        self.frame = frame.copy()
        self.roi = self.make_roi(frame)

    #convert Frame as grayFrame
    def cvGray(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        self.gray = cv2.GaussianBlur(gray, (7, 7), 0)

    ##region of interest
    def make_roi(self, frame):
        roi = frame[self.top:self.bottom, self.right:self.left]
        return roi
    
    def get_frame(self):
        return self.frame
    
    def get_roi(self):
        return self.roi
    
    def get_gray(self):
        return self.gray
    
    def left2top(self):
        return self.left, self.top

    def right2bottom(self):
        return self.right, self.bottom

    


#배경 찍어내기
def run_avg(image, accumWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, accumWeight)


#배경으로 부터 손부분 잘라내기
def segment(image, threshold=7):
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
    

    


        


