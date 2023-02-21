import pickle
#from typing import Self

def model_load(model_path, model_name):
    model = pickle.load(model_path + model_name)
    return model


#-------------------------------------------------------------
import cv2

class VideoCustom():
    top, right, bottom, left = 10, 350, 225, 590

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
        self.gray = cv2. GaussianBlur(gray, (7, 7), 0)

    ##region of interest
    def make_roi(self, frame):
        roi = frame[self.top:self.bottom, self.right:self.left]
        return roi
    
    def get_frame(self):
        return self.frame
    
    def get_roi(self):
        return self.roi


    


        


