"""
CALCULATE HISTOGRAM AND EXTRACT FINGERS.

https://dev.to/amarlearning/finger-detection-and-tracking-using-opencv-and-python-586m
"""


import cv2
import numpy as np

lower = np.array([0,130,77], dtype = "uint8")
upper = np.array([235,173,127], dtype = "uint8")

camera = cv2.VideoCapture(0)

while(True):
    
    (grabbed, frame) = camera.read()
    
    frame = cv2.resize(frame, (400, 400))
    
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
        
    skinMask = cv2.inRange(converted, lower, upper)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)
    
    
    cv2.imshow("Press q to exit", np.hstack([frame, skin]))
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()