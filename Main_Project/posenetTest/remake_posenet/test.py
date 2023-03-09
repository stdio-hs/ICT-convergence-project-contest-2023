import cv2
import time
import argparse #CMD에서 파라미터를 추가하여 실행가능하게 하는 라이브러리
import posenet

#텐서플로우 버전 1을 사용하도록 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() #tensorflow V2의 함수를 비활성화


import os
# 경고 수준을 2단계로 설정(경고 비활성화)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(101, sess)
        output_stride = model_cfg['output_stride']





if __name__ == '__main__':
    main()