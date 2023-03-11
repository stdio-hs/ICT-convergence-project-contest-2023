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
    camWidth = 1280
    camheight = 720
    scale_factor = 0.7125


    with tf.Session() as sess:
        #model_cfg : 모델 설정
        #model_outputs : 모델(레이어 및 오프셋, 연산방식 등) 
        model_cfg, model_outputs = posenet.load_model(101, sess)
        output_stride = model_cfg['output_stride'] 
        #load_model 에서 받아온 output_stride == 16

        """
        model_cfg = {
            'output_stride': output_stride, # 16
            'checkpoint_name': checkpoint_name, # mobilenet_v1_101
        }
        """

        cap = cv2.VideoCapture(0) 
        cap.set(3, camWidth) #set(propID, value) propID = 3 (width)
        cap.set(4, camheight)  #set(propID, value) propID = 4 (camheight)

        start = time.time()
        frame_count = 0

        while True:
            # input_image : predict할 이미지(resize한 이미지)
            # display_image : 사용자에게 보여줄 이미지(resize 하지 않은 원본 이미지)
            # output_scale : 원본 이미지와 변형된 이미지의 비율 ndarray (input_image.size / display_imag.size)
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=scale_factor, output_stride = output_stride
            ) # cap, 0.7125, 16

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                fee_dict = {'image:0' : input_image}
            )












if __name__ == '__main__':
    main()