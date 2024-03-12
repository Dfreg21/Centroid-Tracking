import numpy as np
import cv2

cap= cv2.VideoCapture('1.mp4')

w= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc= cv2.VideoWriter_fourcc(*'mp4v')
out0= cv2.VideoWriter('Out0.mp4', fourcc, 30, (w,h))
out1= cv2.VideoWriter('Out1.mp4', fourcc, 30, (w,h),0)

hsv_bmn= (10,50,50)
hsv_bmx= (50,255,255)

def find_object(im, mask, color):
    cnts, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key= cv2.contourArea)

    x,y,w,h = cv2.boundingRect(c)
    cv2.drawContours(im, cnts, -1, (250,25,0), 2)

    return (round(x+w/2), round(y+h/2))

while (True):
    ret, im= cap.read()
    if ret == False:
        break

    hsv= cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    fhsv= cv2.GaussianBlur(hsv,(29,29),0)
    mb= cv2.inRange(hsv, hsv_bmn, hsv_bmx)
    pb= find_object(im, mb, (255,0,0))

    ret, b = cv2.threshold(mb, 90, 255, cv2.THRESH_BINARY)
    c = cv2.HoughCircles(b, cv2.HOUGH_GRADIENT, 1, 20, param1=10, param2=16, minRadius=1, maxRadius=100)

    if c is not None:
        c = np.uint16(np.around(c))
        for i in c[0, :]:
            cv2.circle(im, (i[0], i[1]), 2, (0, 0, 255), 3)
            cv2.circle(mb, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow('Out0', im)
    cv2.imshow('Out1', mb)
    out0.write(im)
    out1.write(mb)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

