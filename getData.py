import cv2
import numpy as np
import sqlite3
import os


cap = cv2.VideoCapture(0)
# load tv
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # captureDevice = camera

def insertOrUpdate(id, name):
    conn = sqlite3.connect('Data.db')
    cmd = "SELECT * FROM Datas WHERE ID=" + id
    cursor = conn.execute(cmd)

    isRecordExist = 0

    for row in cursor:
        isRecordExist = 1

    if (isRecordExist==0):
        cmd = "INSERT INTO Datas(Id,Name) Values(" + id + "," + name + ")"
    else:
        cmd = "UPDATE Datas SET Name=" + name + " WHERE ID=" + id

    conn.execute(cmd)
    conn.commit()
    conn.close()

# insert to db
id = str(input("Nhập ID: "))
name = str(input("Nhập Name: "))
insertOrUpdate(id, name)

sampleNum = 0

while (True):
    _, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        sampleNum += 1
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite('dataSet/User.' + str(id) + '.' + str(sampleNum) + ' .jpg', gray[y: y + h, x: x + w])

    cv2.imshow('Output', img)
    cv2.waitKey(1)

    if sampleNum > 200:
        break

cap.release()
cv2.destroyAllWindows()





    








