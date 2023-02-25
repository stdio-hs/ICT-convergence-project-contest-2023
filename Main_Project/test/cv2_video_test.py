import cv2


#cap = cv2.VideoCapture(0)
video_file = './sampleVideo.mp4'

cap = cv2.VideoCapture(video_file)

def main():
    if cap.isOpened():
        while True:
            ret, img = cap.read()
            if ret:
                cv2.imshow(video_file, img)
                cv2.waitKey(25)
            else:
                print(ret)
                print('wrong access')
                break

    else:
        print('video can not open')
    cap.release()
    cv2.destroyAllWindows()



def main1():
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    count = 0
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('sampleVideo.mp4', fourcc, fps, (w, h))
    #model = cm.model_load(model_path)

    if not out.isOpened():
        print('File open failed')
        cap.release()
        exit(1)

    while cap.isOpened():
        ret, frame = cap.read()

        if ret is not True:
            print(" >> Camera did not work, check your port or status << ")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2BGR)
        

        cv2.imshow('frame', gray)
        out.write(gray) #영상 데이터 저장, 소리 X

        keypress = cv2.waitKey(1) & 0xFF

        if keypress == ord('q'):
            break

    cap.release()
    out.release() 
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
        

