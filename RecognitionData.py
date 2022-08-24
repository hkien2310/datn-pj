import cv2
import numpy as np
import os
import sqlite3
from PIL import Image

#Tranning hình ảnh nhận diện
#Thư viện nhận diện khuôn mặt
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('recognizer/trainningData.yml')

#get data from sqlite by ID
def getProfile(id):
    conn = sqlite3.connect('Data.db')
    query = "SELECT * FROM Datas WHERE ID="+str(id)
    cursor = conn.execute(query)
    
    profile=None
    
    for row in cursor:
        profile = row
        
    conn.close()
    return profile

cap = cv2.VideoCapture(0)

fontface = cv2.FONT_HERSHEY_SIMPLEX



while(True):
    #camera read
    _, img = cap.read()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray)
    
    for(x, y, w, h) in faces:
        cv2.rectangle(img , (x, y), (x+w, y+h), (0, 255, 0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        
        id, confidence = recognizer.predict(roi_gray)
        print(confidence)
        if(confidence < 50):
            profile = getProfile(id)
            print(profile)
            if(profile != None):
                # cv2.putText(frame, id, (10,30), fontface, 1, (0, 0, 255), 2)
                cv2.putText(img, str(profile[1]), (x+10, y+h+30), fontface, 1, (0,255,0) ,2)
        else:
            cv2.putText(img, "Unkown", (x+10, y+h+30), fontface, 1, (0,0,255) ,2)
        
    cv2.imshow('Output', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()








