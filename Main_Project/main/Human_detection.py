'''
1. 사람을 인식
2. 특정 제스처를 인식(주먹을 쥐는 모션)
3. 제스처 모드로 진입
'''


import cv2
import numpy as np
from PIL import Image
import sys
import common as cm

sys.path.insert(0, './')

cap = cv2.VideoCapture(0)
threshold = 0.2
top_k = 5  #타겟 갯수

model_dir = './'
human_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
lbl = 'coco_labels.txt'

counter = 0
prev_val=0

selected_obj = "person"

def show_selected_object_counter(objs, labels):
    global counter, prev_val, selected_obj

    arr = []
    for obj in objs:
        label = labels.get(obj.id, obj.id)
        arr.append(label)

    x = arr.count(selected_obj)

    diff = x - prev_val

    if (diff>0):
        counter = counter + diff
    
    prev_val = x



def customFrame(cv2_im):
    cv2_im = cv2.flip(cv2_im, 0) #상하 반전
    cv2_im = cv2.flip(cv2_im, 1) #좌우 반전
    cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB) #BGR to RGB
    pil_im = Image.fromarray(cv2_im_rgb)
    return pil_im

def main():

    mdl = human_model
    interpreter, labels = cm.load_model(model_dir,mdl,lbl)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("cap.read() does not work well")
            break

        pil_im = customFrame(frame) #frame 변형

        cm.set_input(interpreter, pil_im)
        interpreter.invoke() #연산 권한을 위임
        objs = cm.get_output(interpreter, score_threshold = threshold, top_k = top_k)

        show_selected_object_counter(objs, labels) #라벨 및 탐지된 사물 갯수 체크

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2_im = cm.append_text_img(cv2_im, objs, labels, counter, selected_obj)

    cap.release() #release memories
    cv2.destroyAllWindows() #close winodws 

if __name__ == '__main' :
    main()

















