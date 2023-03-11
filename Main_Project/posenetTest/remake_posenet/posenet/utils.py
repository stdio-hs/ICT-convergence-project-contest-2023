import cv2
import numpy as np

def valid_resolution(width, height, output_stride = 16):
    target_width = (int(width) // output_stride) * output_stride + 1
    target_height = (int(height) // output_stride) * output_stride + 1

    #print('target_height :', target_height)
    #print('target_width:', target_width)

    return target_height, target_width


def read_cap(cap, scale_factor = 1.0, output_stride = 16):
    ret, img = cap.read()

    # 예외처리
    if not ret:
        raise IOError("webcam failure")

    # print('img width', img.shape[1])
    # print('transed width', img.shape[1] * scale_factor)

    # 원본 이미지 크기에서 predict에 사용할 이미지 크기를 뽑아냄
    target_width, target_height = valid_resolution(
        img.shape[1] * scale_factor, img.shape[0]  * scale_factor, output_stride=output_stride
        )

    # 원본과의 비율을 저장한 ndarray
    scale = np.array([img.shape[1] / target_height, img.shape[1] / target_width])

    # predict에 사용할 크기로 resize함 (INTER_LINEAR : 양선형 보강법)
    input_img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    # cv2에서 videocapture로 들어온 프레임은 BGR 형태로 저장되므로 RGB형태로 바꾸어 연산이 가능하게 함
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32) # 타입은 float형태
    # 이거는 어디에 쓰는 걸까요?

    #print(input_img)
    #print(input_img * (2.0 / 255.0) - 1.0)

    input_img = input_img * (2.0 / 255.0) - 1.0
    #cv2.imshow('utils', input_img) #아주 어둡게 출력되는 것을 알 수 있다. 왜?
    #실험 결과 곱하는 값이 크면 클수록 밝에 나오는데, 2.0 대신 10을 넣으면 아주 약간만 인식된다.
    #반대로 어두운 것은 상관없이 얼마나 어둡든 잘 찍혀나온다는 것을 알 수 있었다.

    # model의 차원에 맞추어 4차원 형태로 변형
    input_img = input_img.reshape(1, target_height, target_width, 3)

    # predict에 사용할 이미지, 원본이미지, 비율
    return input_img, img, scale