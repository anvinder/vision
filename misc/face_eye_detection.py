import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier ('face_detection_haar_cascade.xml')
eye_cascade  = cv2.CascadeClassifier ('eye_detection_haar_cascade.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face= face_cascade.detectMultiScale(gray, 1.3, 5)
    for (l,b,w,h) in face:
        cv2.rectangle(frame, (l,b),(l+w, b+h), (255,0,0), 2)
        roi_gray =  gray[b:b+h, l:l+w ]
        roi_color = frame[b:b+h, l:l+w]
        eye = eye_cascade.detectMultiScale(roi_gray)
        for (el, eb, ew, eh) in eye:
            cv2.rectangle(roi_color, (el, eb), (el+ew, eb+eh), (0,255,0), 2)

    cv2.imshow('haar features', frame)
    k=cv2.waitKey(30) & 0xFF
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()
