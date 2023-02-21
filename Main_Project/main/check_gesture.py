import cv2
import numpy as np
import os

import matplotlib.pyplot as plt
import glob
import common2 as cm

model_path = ('./../test/')
model_name = 'model.pt'


def main():
    cap = cv2.VideoCapture(0)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    num_frames = 0

    #model = cm.model_load(model_path, model_name)

    while cap.isOpened():
        ret, frame = cap.read()

        #정상 동작하는지 체크
        if ret is not True:
            print(" >> Camera did not work, check your port or status << ")
            break

        cv_frame = cm.VideoCustom(frame)
    
        #화면에 송출
        cv2.imshow('Captured Frame : ', cv_frame.get_frame())


        # q를 누르면 종료 
        if cv2.waitKey(1) == ord('q'):
            break
        keypress = cv2.waitKey(1) & 0xFF

    #cv2 프레임 메모리 반환
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
   main()