"""
import cv2
# Starts capturing video
cap = cv2.VideoCapture(0)

fps = int(cap.get(cv2.CAP_PROP_FPS))

print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

while cap.isOpened():
    ret, frame = cap.read()
    cv2.imshow('Captured Frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

    keypress = cv2.waitKey(1) & 0xFF

cap.release()
cv2.destroyAllWindows()
"""


import cv2
import platform
src = 0

captrue = cv2.VideoCapture( src )
captrue.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
captrue.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
while captrue.isOpened():
    
    (grabbed, frame) = captrue.read()
    
    if grabbed:
        cv2.imshow('Wandlab Camera Window', frame)
 
        key = cv2.waitKey(1) & 0xFF
        if (key == 27): 
            break
 
captrue.release()
cv2.destroyAllWindows()