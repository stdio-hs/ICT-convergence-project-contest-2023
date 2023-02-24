import cv2
import os

import matplotlib.pyplot as plt
import glob
import common2 as cm



model_path = ('./../../../finger_model1/')


def main():
    cap = cv2.VideoCapture(0)
    k = 0
    accumWeight = 0.1
    num_frames = 0

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    
    model = cm.model_load(model_path)
    #model = cm.model_load(model_path, model_name)
    
    while cap.isOpened():
        ret, frame = cap.read()

        #정상 동작하는지 체크
        if ret is not True:
            print(" >> Camera did not work, check your port or status << ")
            break

        cv_frame = cm.VideoCustom(frame, fps)


        #처음 30프레임은 배경을 위해 버림
        if num_frames < 30:
            cv_frame.run_avg(accumWeight)
            #cm.run_avg(cv_frame.get_gray(), accumWeight)
            if num_frames == 1:
                print("[STATUS] please wait! calibrating...")
            elif num_frames == 29:
                print("[STATUS] calibration successfull...")
        
        else:
            hand = cv_frame.segment()
            #hand = cm.segment(cv_frame.get_gray())
            # 여기 이프 부분 common2로 이동시키고, 클래스에서 통합관리 시키기
            if hand is not None:
                cm.roi_workin(hand, cv_frame, model, k)

        k += 1
        #손이 들어가는 네모 그리기
        top, right, bottom, left = cv_frame.getSize()
        cv2.rectangle(cv_frame.clone, (left, top), (right, bottom), (0, 255, 0), 2)
        #프레임 수 체크
        num_frames += 1

        #화면에 송출
        cv2.imshow("Video Feed", cv_frame.clone)

        keypress = cv2.waitKey(1) & 0xFF

        if keypress == ord("q"):
            break

    #cv2 프레임 메모리 반환
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
   main()