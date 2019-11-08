"""
CALCULATE HISTOGRAM AND EXTRACT FINGERS.

https://dev.to/amarlearning/finger-detection-and-tracking-using-opencv-and-python-586m

Implement save histogram using a key press 
"""


import cv2
import numpy as np
import math

face_cascade = cv2.CascadeClassifier('Tools/haarcascade_frontalface_default.xml')
lower = np.array([0,130,77], dtype = "uint8")
upper = np.array([235,173,127], dtype = "uint8")

camera = cv2.VideoCapture(0)

def draw_rect(frame):
    rows, cols, _ = frame.shape
    global total_rectangle, hand_rect_one_x, hand_rect_one_y, hand_rect_two_x, hand_rect_two_y

    hand_rect_one_x = np.array(
        [6 * rows / 20, 6 * rows / 20, 6 * rows / 20, 9 * rows / 20, 9 * rows / 20, 9 * rows / 20, 12 * rows / 20,
         12 * rows / 20, 12 * rows / 20], dtype=np.uint32)

    hand_rect_one_y = np.array(
        [9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20,
         10 * cols / 20, 11 * cols / 20], dtype=np.uint32)

    hand_rect_two_x = hand_rect_one_x + 10
    hand_rect_two_y = hand_rect_one_y + 10
    total_rectangle = 9
    for i in range(total_rectangle):
        cv2.rectangle(frame, (hand_rect_one_y[i], hand_rect_one_x[i]),
                      (hand_rect_two_y[i], hand_rect_two_x[i]),
                      (0, 255, 0), 1)

    return frame

def hand_histogram(frame):
    global hand_rect_one_x, hand_rect_one_y

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)

    for i in range(total_rectangle):
        roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[hand_rect_one_x[i]:hand_rect_one_x[i] + 10,
                                          hand_rect_one_y[i]:hand_rect_one_y[i] + 10]

    hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)
    
def hist_masking(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h = (h +180) % 180
    s_del = s < 50
    s[s_del] = 0
    hsv = cv2.merge([h, s, v])
    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    cv2.filter2D(dst, -1, disc, dst)

    ret, thresh = cv2.threshold(dst, 30, 255, cv2.THRESH_BINARY)
    
    thresh = cv2.merge((thresh, thresh, thresh))
    
    return cv2.bitwise_and(frame, thresh)
    
def connected_components(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    for (x,y,w,h) in faces:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),-2)
    ret, labels = cv2.connectedComponents(frame, connectivity = 8)
    label_hue = np.uint8(179*labels/np.max(labels))
   
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([blank_ch, blank_ch, blank_ch])
   
    labeled_img[label_hue==0] = 0
    return labeled_img

    
def convex_hull(frame_1, frame):

    frame = cv2.GaussianBlur(frame,(5,5),100)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(frame, 30, 255, cv2.THRESH_BINARY)
    contours,hierarchy= cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key = lambda x: cv2.contourArea(x))
    epsilon = 0.0005*cv2.arcLength(cnt,True)
    approx= cv2.approxPolyDP(cnt,epsilon,True)
    hull = cv2.convexHull(cnt)
    areahull = cv2.contourArea(hull)
    areacnt = cv2.contourArea(cnt)
    arearatio=((areahull-areacnt)/areacnt)*100
    hull = cv2.convexHull(approx, returnPoints=False)
    defects = cv2.convexityDefects(approx, hull)
    
    l=0
    frame_x = frame_1.copy()
    for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            pt= (100,180)
            
            
            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            s = (a+b+c)/2
            ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
            
            #distance between point and convex hull
            d=(2*ar)/a
            
            # apply cosine rule here
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
            
        
            # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
            if angle <= 90 and d>30:
                l += 1
                cv2.circle(frame_x, far, 3, [255,0,0], -1)
            
            #draw lines around hand
            cv2.line(frame_x,start, end, [0,255,0], 2)
            
    l += 1
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    if l==1:
        if areacnt<2000:
            cv2.putText(frame_x,'Put hand in the box',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        else:
            if arearatio<12:
                cv2.putText(frame_x,'0',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            elif arearatio<17.5:
                cv2.putText(frame_x,'Best of luck',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                   
            else:
                cv2.putText(frame_x,'1',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    
    elif l==2:
        cv2.putText(frame_x,'2',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
    elif l==3:
         
        if arearatio<27:
            cv2.putText(frame_x,'3',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        else:
            cv2.putText(frame_x,'ok',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    
    elif l==4:
        cv2.putText(frame_x,'4',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
    elif l==5:
        cv2.putText(frame_x,'5',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
    elif l==6:
        cv2.putText(frame_x,'reposition',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
    else :
        cv2.putText(frame_x,'reposition',(10,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
    
    return frame_x


if __name__ == '__main__':
    
    sample_grabbed = False
    while(True):
    
        (grabbed, frame) = camera.read()
    
        frame = cv2.resize(frame, (400, 400))
    
        if cv2.waitKey(1) & 0xFF == ord("z"):
            sample_grabbed = True
    
        if sample_grabbed is False:
            frame = draw_rect(frame)
            hist = hand_histogram(frame)
    
        backprojected_image = hist_masking(frame, hist)
        connected_comps = connected_components(backprojected_image)
        post_convex_hull = convex_hull(frame, connected_comps)
        cv2.imshow("Press q to exit", np.hstack([frame ,backprojected_image, post_convex_hull, connected_comps]))
    
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()