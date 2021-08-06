import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math

#hand detector instance
detector = htm.HandDetector(detection_confidence=0.7)

#camera stream
camera_width, camera_height = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, camera_width)
cap.set(4, camera_width)

#to calc fps
pTime = 0
cTime = 0

while True:
    #Grab frame
    success, img = cap.read()

    #Detect Hand landmarls
    img = detector.find_hands(img)
    landmark_list = detector.find_hand_position(img, draw=False)

    if (len(landmark_list) != 0):

        #Index finger and thumb coordiantes
        x1, y1 = landmark_list[4][1], landmark_list[4][2]
        x2, y2 = landmark_list[8][1], landmark_list[8][2]
        #Center betweent he finger and the thumb
        cx, cy = ((x1 + x2) // 2) , ((y1 + y2) // 2)

        #Circles and line
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (cx, cy), 10, (0,255,255), cv2.FILLED)
        cv2.line(img, (x1 , y1), (x2, y2), (255, 0, 255), 3)
        
        #Make red green when fingers close enough
        length = math.hypot(x2-x1,y2-y1)
        if length < 50:
            cv2.circle(img, (cx, cy), 10, (0,0,255), cv2.FILLED)

    #Calc fps
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    #Draw fps
    cv2.putText(img, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
    
    #Show image
    cv2.imshow("Img", img)
    cv2.waitKey(1)
