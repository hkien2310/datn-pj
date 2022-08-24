from tkinter import *
from tkinter.ttk import *
import tkinter
import cv2
import PIL.Image,PIL.ImageTk
from djitellopy import tello
import sqlite3
import Face
import numpy as np

# datn = tello.Tello()
# datn.connect()
# print(datn.get_battery())
# datn.streamon()
# speed = 20

window = Tk()
window.title("Giao diện")
window.geometry("800x600")

imgbg = cv2.imread('background.jpg')
imgbg = cv2.cvtColor(imgbg, cv2.COLOR_BGR2RGB)
imgbg = cv2.resize(imgbg, (800, 600))
imgbg = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(imgbg))

imglogo = cv2.imread('logohaui.png')
imglogo = cv2.cvtColor(imglogo, cv2.COLOR_BGR2RGB)
imglogo = cv2.resize(imglogo, (100,100))
imglogo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(imglogo))

btn1, btn2, btn3, btn4, btnu, btnd, btnf, btnb, btnl, btnr, btny, btnv,\
btnm, btna, btng = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
def FindFace():
    global btn1
    btn1 = 1 - btn1
def FaceMesh():
    global btn2
    btn2 = 1 - btn2
def FindHands():
    global btn3
    btn3 = 1 - btn3
def FindPose():
    global btn4
    btn4 = 1 - btn4
def Up():
    global btnu
    btnu = 1 - btnu
def Down():
    global btnd
    btnd = 1 - btnd
def Forward():
    global btnf
    btnf = 1 - btnf
def Behind():
    global btnb
    btnb = 1 - btnb
def Left():
    global btnl
    btnl = 1 - btnl
def Right():
    global btnr
    btnr = 1 - btnr
def RotateL():
    global btny
    btny = 1 - btny
def RotateR():
    global btnv
    btnv = 1 - btnv
def Manual():
    global btnm
    btnm = 1 - btnm
def Auto():
    global btna
    btna = 1 - btna
def Gesture():
    global btng
    btng = 1 - btng

video = cv2.VideoCapture(0)
bg = Label(window, image=imgbg)
bg.place(x=0, y=0)
# lbl = Label(window, text="Camera", font=("Arial",20))
# lbl.pack()
canvas = Canvas(window, width= 500, height=400)
canvas.place(x=150, y=0)
lg = Label(window, image=imglogo)
lg.place(x=0, y=0)
button1 = tkinter.Button(window, text='Tìm mặt', activebackground='red', width=7, height=1, command=FindFace)
button1.place(x=210, y=413)
button2 = tkinter.Button(window, text='Mặt nạ', activebackground='red', width=7, height=1, command=FaceMesh)
button2.place(x=310, y=413)
button3 = tkinter.Button(window, text='Tìm tay', activebackground='red', width=7, height=1, command=FindHands)
button3.place(x=410, y=413)
button4 = tkinter.Button(window, text='Tìm dáng', activebackground='red', width=7, height=1, command=FindPose)
button4.place(x=510, y=413)
buttonu = tkinter.Button(window, text='Up', width=7, height=1, command=Up)
buttonu.place(x=250, y=458)
buttond = tkinter.Button(window, text='Down', width=7, height=1, command=Down)
buttond.place(x=250, y=483)
buttonl = tkinter.Button(window, text='Left', width=7, height=1, command=Left)
buttonl.place(x=430, y=483)
buttonr = tkinter.Button(window, text='Right', width=7, height=1, command=Right)
buttonr.place(x=550, y=483)
buttonf = tkinter.Button(window, text='FW', width=7, height=1, command=Forward)
buttonf.place(x=490, y=458)
buttonb = tkinter.Button(window, text='BW', width=7, height=1, command=Behind)
buttonb.place(x=490, y=483)
buttonrl = tkinter.Button(window, text='RotateL', width=7, height=1, command=RotateL)
buttonrl.place(x=190, y=483)
buttonrr = tkinter.Button(window, text='RotateR', width=7, height=1, command=RotateR)
buttonrr.place(x=310, y=483)
buttonm = tkinter.Button(window, text='Manual', activebackground='red', width=7, height=1, command=Manual)
buttonm.place(x=670, y=150)
buttona = tkinter.Button(window, text='Auto', activebackground='red', width=7, height=1, command=Auto)
buttona.place(x=670, y=200)
buttong = tkinter.Button(window, text='Gesture', activebackground='red', width=7, height=1, command=Gesture)
buttong.place(x=670, y=250)

detector = Face.FaceDetection()
detector1 = Face.FaceMesh()
detector2 = Face.Hand()
detector3 = Face.Pose()

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

def faceRecognition(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    name = []
    cx=[]
    for (x, y, w, h) in faces:
        #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cx = x + w // 2
        cy = y + h // 2
        #cv2.circle(img, (cx, cy), 5,  (0, 0, 255), cv2.FILLED)

        roi_gray = gray[y:y + h, x:x + w]

        id, confidence = recognizer.predict(roi_gray)
        #print(confidence)
        if (confidence < 90):
            profile = getProfile(id)
            #print(profile)
            if (profile != None):
                # cv2.putText(frame, id, (10,30), fontface, 1, (0, 0, 255), 2)
                # cv2.putText(img, str(profile[1]), (x + 10, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                name = profile[1]
        else:
            # cv2.putText(img, "Unkown", (x + 10, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            name = 0

    return img, name, cx

pid = [0.4, 0.4, 0]
pError = 0
def trackFace(cx, pid, pError):
    error = cx - 250
    spd = pid[0]*error + pid[1]*(error-pError)
    spd = int(np.clip(spd, -100, 100))

    if cx==0:
        spd = 0
        error = 0
    print(spd)
    #datn.send_rc_control(0, 0, 0, spd)

    return error

def Control():
    if btnu == 1:
        print("Up")
        #datn.send_rc_control(0, 0, speed, 0)
    if btnd == 1:
        print("Down")
        #datn.send_rc_control(0, 0, -speed, 0)
    if btny == 1:
        print("Rotate Left")
        #datn.send_rc_control(0, 0, 0, -speed)
    if btnv == 1:
        print("Rotate Right")
        #datn.send_rc_control(0, 0, 0, speed)
    if btnf == 1:
        print("Forward")
        #datn.send_rc_control(0, speed, 0, 0)
    if btnb == 1:
        print("Backward")
        #datn.send_rc_control(0, -speed, 0, 0)
    if btnl == 1:
        print("Left")
        #datn.send_rc_control(-speed, 0, 0, 0)
    if btnr == 1:
        print("Right")
        #datn.send_rc_control(speed, 0, 0, 0)

def update_frame():
    global canvas, photo, pError
    _, img = video.read()
    #img = datn.get_frame_read().frame
    img = cv2.resize(img, dsize=None, fx=0.8, fy=0.85)
    if btnm==1:
        print("K có j đâu, bấm nút cũng vô ích thôi")
        Control()
    if btna==1:
        img, name, cx = faceRecognition(img)
        if name=="luu":
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = detector3.findPose(img)
            lmList3, bbox3 = detector3.findPosition(img)
            if len(lmList3) != 0:
              detector3.ControlDrone(img, 15, 11, 23, 16, 12, 24, 0)
        else:
            print("Nothing")
    if btng==1:
        img = detector2.findHands(img)
        lmList2, bbox2 = detector2.findPosition(img)
        detector2.ControlTello(lmList2)

    if btn1==1:
        img, name, cx = faceRecognition(img)
        pError = trackFace(cx, pid, pError)
        img = detector.findFaces(img)
    if btn2==1:
        img = detector1.findFaceMesh(img)
    if btn3==1:
        img = detector2.findHands(img)
        lmList2, bbox2 = detector2.findPosition(img)
    if btn4==1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = detector3.findPose(img)
        lmList3, bbox3 = detector3.findPosition(img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0,0, image = photo, anchor=tkinter.NW)
    window.after(15, update_frame)

update_frame()
window.mainloop()

# datn.streamoff()
video.release()
cv2.destroyAllWindows()
