import pickle
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array, array_to_img
from sklearn import preprocessing
#from typing import Self
labels = ['blank', 'fist', 'five', 'ok', 'thumbsdown', 'thumbsup']
f_label = ['0L', '0R', '1L', '1R', '2L', '2R', '3L', '3R', '4L', '4R', '5L', '5R',]

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
    #cv2.resize(frame, (128, 128))
    gray_image = img_to_array(gray_image)
    gray_image = np.expand_dims(gray_image, axis = 0)
    prediction = model.predict(gray_image)
    return prediction

def _getLabel(predict):
    print(predict)
    arg_class = np.argmax(predict)
    print(arg_class)
    predict_label = labels[arg_class]
    return predict_label




#-------------------------------------------------------------
import cv2

#글로벌 변수 (배경)
bg = None


#cv2로 받은 비디오를 총괄하여 관리하는 클래스
#240 215
class VideoCustom():
    #top, right, bottom, left = 30, 200, 130, 320
    top, right, bottom, left = 30, 200, 270, 415
    # finger set 
    #top, right, bottom, left = 30, 200, 158, 328
    #gray, frame, roi

    def __init__(self, frame, fps):
        self.customFrame(frame)
        self.cvGray()
        self.fps = fps

    def getSize(self):
        return self.top, self.right, self.bottom, self.left
        
    def customFrame(self, frame):
        #프레임 크기 변형
        self.frame = frame
        #self.frame = cv2.resize(frame, (700, 700))
        #프레임 좌우 반전
        #self.frame = cv2.flip(frame, 1)
        self.clone = self.frame.copy()
        (self.height, self.width) = frame.shape[:2]
        self.make_roi(frame)

    #convert Frame as grayFrame
    def cvGray(self):
        gray = cv2.cvtColor(self.roi, cv2.COLOR_BGR2GRAY)
        self.gray = cv2.GaussianBlur(gray, (0, 0), 1 )
        #self.gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        

    ##region of interest
    def make_roi(self, frame):
        self.roi = frame[self.top:self.bottom, self.right:self.left]

    #배경 찍어내기
    def run_avg(self, accumWeight):
        global bg
        # initialize the background
        if bg is None:
            bg = self.gray.copy().astype("float")
            return

        # compute weighted average, accumulate it and update the background
        cv2.accumulateWeighted(self.gray, bg, accumWeight)
        return 

    #배경으로 부터 손부분 잘라내기
    def segment(self, threshold=15):
        global bg
        # find the absolute difference between background and current frame
        #print(self.bg)
        if bg is not None:
            cv2.imshow("gray", self.gray)
            diff = cv2.absdiff(bg.astype("uint8"), self.gray)
            temp = np.array(diff)
            #print(temp.mean())

            # 차이가 threshold 보다 크면 흰색, threshold보다 작으면 검은색
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
        
        return
    

def roi_workin(hand, cv_frame, model, k):
    (thresholded, segmented) = hand

    #roi 내부의 외각선 그리기
    cv2.drawContours(cv_frame.clone, [segmented + (cv_frame.right, cv_frame.top)], -1, (0, 0, 255))
    #print('k = ', k)
    #print('fps = ', cv_frame.fps)
    

    if k % (cv_frame.fps // 6) == 0:
        #스레드홀드 temp.png로 보관
        cv2.imwrite('Temp.png', thresholded)
        #예측
        predictedClass = getPredict(model, thresholded)
        print(predictedClass)
        cv2.putText(cv_frame.clone, str(predictedClass), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 스레드 홀드 보여주기
    cv2.imshow("Thesholded", thresholded)






        


