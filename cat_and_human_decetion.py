# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 17:20:02 2023

@author: Tako
"""

import cv2
import matplotlib.pyplot as plt
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cat_face_cascade = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")


cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()
   
    if ret:
        face_rect = face_cascade.detectMultiScale(frame, minNeighbors = 7)
        
        
        for (x,y,w,h) in face_rect:
            cv2.rectangle(frame, (x,y),(x+w, y+h),(0,255,0),3)           
            cv2.putText(frame,"Human Detected",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255),2)
        #cv2.imshow("face detect", frame)
        
        cat_face_rect = cat_face_cascade.detectMultiScale(frame, minNeighbors = 7)
        
        for (x,y,w,h) in cat_face_rect:
            cv2.rectangle(frame, (x,y),(x+w, y+h),(0,255,0),3)            
            cv2.putText(frame,"Cat Detected",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255),2)
        cv2.imshow("face detect", frame)

        
        
    
    if cv2.waitKey(1) & 0xFF == ord("q"): break
cap.release()
cv2.destroyAllWindows()


















