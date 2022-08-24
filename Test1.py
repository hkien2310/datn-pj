import cv2
import numpy as np
import os
import sqlite3
from PIL import Image

# Tranning hình ảnh nhận diện
# Thư viện nhận diện khuôn mặt
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('recognizer/trainningData.yml')


# get data from sqlite by ID
def getProfile(id):
    conn = sqlite3.connect('Data.db')
    query = "SELECT * FROM Datas WHERE ID=" + str(id)
    cursor = conn.execute(query)

    profile = None

    for row in cursor:
        profile = row

    conn.close()
    return profile


cap = cv2.VideoCapture(0)

fontface = cv2.FONT_HERSHEY_SIMPLEX

pid = [0.4, 0.4, 0]
pError = 0
def trackFace(cx, pid, pError):
    x = cx
    error = x - 250
    speed = pid[0]*error + pid[1]*(error-pError)
    speed = int(np.clip(speed, -100, 100))

    if x==0:
        speed = 0
        error = 0
    print(speed)

    return error

while (True):
    # camera read
    _, img = cap.read()
    img = cv2.resize(img,(500,400))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray)
    center = []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cx = x + w // 2
        cy = y + h // 2
        cv2.circle(img, (cx, cy), 5,  (0, 0, 255), cv2.FILLED)


        roi_gray = gray[y:y + h, x:x + w]

        id, confidence = recognizer.predict(roi_gray)
        #print(confidence)
        if (confidence < 90):
            profile = getProfile(id)
            #print(profile)
            if (profile != None):
                # cv2.putText(frame, id, (10,30), fontface, 1, (0, 0, 255), 2)
                cv2.putText(img, str(profile[1]), (x + 10, y + h + 30), fontface, 1, (0, 255, 0), 2)
        else:
            cv2.putText(img, "Unkown", (x + 10, y + h + 30), fontface, 1, (0, 0, 255), 2)
    pError = trackFace(cx, pid, pError)


    cv2.imshow('Output', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()








