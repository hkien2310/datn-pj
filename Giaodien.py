#Khai báo thư viện
from tkinter import *
import math
from tkinter.ttk import *
import tkinter
import cv2
import matplotlib.pyplot as plt
import PIL.Image,PIL.ImageTk
from djitellopy import tello
import sqlite3
import Face
import numpy as np

video = cv2.VideoCapture(0)

# #Kết nối với Drone
# datn = tello.Tello()
# datn.connect()
# print(datn.get_battery()) # Hiển thị % pin hiện tại
# # Lệnh cất cánh
# datn.takeoff()
# #Lệnh bật camera của drone
# datn.streamon()
# speed = 20

#Tạo cửa sổ giao diện với thư viện tkinter
window = Tk()
window.title("Giao diện")
window.geometry("800x600")

# Đọc ảnh Background
imgbg = cv2.imread('background.jpg')
imgbg = cv2.cvtColor(imgbg, cv2.COLOR_BGR2RGB)
imgbg = cv2.resize(imgbg, (800, 600))
imgbg = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(imgbg))

#Đọc ảnh Logo HaUI
imglogo = cv2.imread('logohaui.png')
imglogo = cv2.cvtColor(imglogo, cv2.COLOR_BGR2RGB)
imglogo = cv2.resize(imglogo, (100,100))
imglogo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(imglogo))

#Khai báo các biến nút bấm
btn1, btn2, btn3, btn4, btnu, btnd, btnf, btnb, btnl, btnr, btny, btnv, btns,\
btnm, btna, btng = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

#Các chương trình của nút bấm
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
def Stop():
    global btns
    btns = 1 - btns

#Thao tác với cửa sổ
#Hiển thị Background trên cửa sổ
bg = Label(window, image=imgbg)
bg.place(x=0, y=0)
# lbl = Label(window, text="Camera", font=("Arial",20))
# lbl.pack()
#Tạo canvas trên cửa sổ
canvas = Canvas(window, width= 500, height=400)
canvas.place(x=150, y=0)
#Hiển thị Logo trên cửa sổ
lg = Label(window, image=imglogo)
lg.place(x=0, y=0)
#Tạo nút bấm trên cửa sổ
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
buttons = tkinter.Button(window, text='Stop', width=7, height=1, command=Stop)
buttons.place(x=650, y=483)
buttonm = tkinter.Button(window, text='Manual', activebackground='red', width=7, height=1, command=Manual)
buttonm.place(x=670, y=150)
buttona = tkinter.Button(window, text='Auto', activebackground='red', width=7, height=1, command=Auto)
buttona.place(x=670, y=200)
buttong = tkinter.Button(window, text='Gesture', activebackground='red', width=7, height=1, command=Gesture)
buttong.place(x=670, y=250)


# Tranning hình ảnh nhận diện
# Thư viện nhận diện khuôn mặt
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('recognizer/trainningData.yml')

# Lấy dữ liệu từ SQLite3
def getProfile(id):
    conn = sqlite3.connect('Data.db')
    query = "SELECT * FROM Datas WHERE ID=" + str(id)
    cursor = conn.execute(query)

    profile = None

    for row in cursor:
        profile = row

    conn.close()
    return profile

a = []
#Nhận diện khuôn mặt
def faceRecognition(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    name = []
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]

        id, confidence = recognizer.predict(roi_gray)
        print(confidence)
        a.append(int(confidence))
        if (confidence < 80):
            profile = getProfile(id)
            #print(profile)
            if (profile != None):
                # cv2.putText(frame, id, (10,30), fontface, 1, (0, 0, 255), 2)
                cv2.putText(img, str(profile[1]), (x + 10, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                name = profile[1]
        else:
            cv2.putText(img, "Unkown", (x + 10, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            name = 0


    return img, name

pid = [0.4, 0.4, 0]
pError0, pError1 = 0, 0
area = 0
Range = [20000,30000]
t = 0.7
def trackFace(bbox, pid, pError0, pError1): #Chương trình bám theo khuôn mặt
    x,y,w,h = bbox
    cx = x+w//2
    cy = y+h//2
    area = w*h
    # print(area)
    fb = 0
    if area < Range[0] and area !=0:
        fb = 10
    if area > Range[1]:
        fb = -10
    if Range[0] < area < Range[1] and area == 0:
        fb = 0

    error0 = cx - 250
    yv = pid[0]*error0 + pid[1]*(error0-pError0)
    yv = int(np.clip(yv, -100, 100))
    yv = int(yv * t)

    error1 = cy - 200
    ud = pid[0]*error1 + pid[1]*(error1-pError1)
    ud = int(np.clip(ud, -100, 100))
    ud = int(-ud * t)


    if cx==0:
        yv = 0
        error0 = 0
    if cy==0:
        ud = 0
        error1 = 0
    print(yv,ud)
    # datn.send_rc_control(0, fb, ud, yv)

    return error0, error1

#Chương trình điển khiển Drone với nút bấm
def Control():
    if btnu == 1:
        print("Up")
        # datn.send_rc_control(0, 0, speed, 0)
    if btnd == 1:
        print("Down")
        # datn.send_rc_control(0, 0, -speed, 0)
    if btny == 1:
        print("Rotate Left")
        # datn.send_rc_control(0, 0, 0, -speed)
    if btnv == 1:
        print("Rotate Right")
        # datn.send_rc_control(0, 0, 0, speed)
    if btnf == 1:
        print("Forward")
        # datn.send_rc_control(0, speed, 0, 0)
    if btnb == 1:
        print("Backward")
        # datn.send_rc_control(0, -speed, 0, 0)
    if btnl == 1:
        print("Left")
        # datn.send_rc_control(-speed, 0, 0, 0)
    if btnr == 1:
        print("Right")
        # datn.send_rc_control(speed, 0, 0, 0)
    if btns == 1:
        print("Stop")
        # datn.send_rc_control(0, 0, 0, 0)

def ControlDrone(img, lmList, p1, p2, p3, p4, p5, p6, p7, draw=True): #Điều khiển Drone bằng cử chỉ cơ thể
    x1, y1 = lmList[p1][1:]
    x2, y2 = lmList[p2][1:]
    x3, y3 = lmList[p3][1:]
    x4, y4 = lmList[p4][1:]
    x5, y5 = lmList[p5][1:]
    x6, y6 = lmList[p6][1:]
    x7, y7 = lmList[p7][1:]

    cacu1 = math.degrees(math.atan2(y2-y1,x2-x1)-math.atan2(y2-y3,x2-x3))
    cacu2 = math.degrees(math.atan2(y5-y4,x5-x4)-math.atan2(y5-y6,x5-x6))
    if cacu1<0:
        cacu1 += 360
    if cacu2<0:
        cacu2 += 360

    if 250<cacu1<290:
        print("Left")
        # datn.send_rc_control(-speed,0,0,0)

    if 80<cacu2<120:
        print("Right")
        # datn.send_rc_control(speed, 0, 0, 0)

    if 80<cacu1<120:
        print("Backward")
        # datn.send_rc_control(0, -speed, 0, 0)

    if 250<cacu2<290:
        print("Forward")
        # datn.send_rc_control(0, speed, 0, 0)

    if 170<cacu1<200:
        print("Down")
        # datn.send_rc_control(0, 0, -speed, 0)

    if 150<cacu2<180:
        print("Up")
        # datn.send_rc_control(0, 0, speed, 0)

    if draw:
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.line(img, (x2, y2), (x3, y3), (255, 255, 255), 2)
        cv2.line(img, (x4, y4), (x5, y5), (255, 255, 255), 2)
        cv2.line(img, (x5, y5), (x6, y6), (255, 255, 255), 2)
        cv2.circle(img, (x1, y1), 10, (0, 255, 0), 2)
        cv2.circle(img, (x2, y2), 10, (0, 255, 0), 2)
        cv2.circle(img, (x3, y3), 10, (0, 255, 0), 2)
        cv2.circle(img, (x4, y4), 10, (0, 255, 0), 2)
        cv2.circle(img, (x5, y5), 10, (0, 255, 0), 2)
        cv2.circle(img, (x6, y6), 10, (0, 255, 0), 2)
        cv2.circle(img, (x7, y7), 10, (0, 255, 0), 2)
        cv2.putText(img, str(int(cacu1)), (x2 - 20, y2 - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.putText(img, str(int(cacu2)), (x5 - 20, y5 - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    # print("cacu1 =", cacu1)
    # print("cacu2 =", cacu2)

def ControlTello(lmList): # Điều khiển drone bằng bàn tay
    Ids = [4, 8, 12, 16, 20]
    # Nếu danh sách điểm trên tay khác 0
    if len(lmList) != 0:
        fingers = []
        # Ngón cái
        if lmList[Ids[0]][1] > lmList[Ids[0] - 2][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 ngón còn lại
        for id in range(1, 5):
            if lmList[Ids[id]][2] < lmList[Ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        totalFingers = fingers.count(1)
        print(totalFingers)
        # if totalFingers == 0:
        #     datn.send_rc_control(0, 0, 0, 0)
        # if totalFingers == 1:
        #     datn.send_rc_control(0, speed, 0, 0)
        # if totalFingers == 2:
        #     datn.send_rc_control(0, -speed, 0, 0)
        # if totalFingers == 3:
        #     datn.send_rc_control(speed, 0, 0, 0)
        # if totalFingers == 4:
        #     datn.send_rc_control(-speed, 0, 0, 0)


detector = Face.FaceDetection()
detector1 = Face.FaceMesh()
detector2 = Face.Hand()
detector3 = Face.Pose()

def update_frame():
    global canvas, photo, pError0, pError1
    _, img = video.read()
    # img = datn.get_frame_read().frame
    img = cv2.resize(img, dsize=None, fx=0.8, fy=0.85)
    if btnm==1:
        # print("K có j đâu, bấm nút cũng vô ích thôi")
        Control()
    if btna==1:
        img, name = faceRecognition(img)
        if name=="luu":
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = detector3.findPose(img)
            lmList3, bbox3 = detector3.findPosition(img)
            if len(lmList3) != 0:
              ControlDrone(img, lmList3, 15, 11, 23, 16, 12, 24, 0)
        else:
            print("Unknown")

    if btng==1:
        img = detector2.findHands(img)
        lmList2, bbox2 = detector2.findPosition(img)
        ControlTello(lmList2)

    if btn1==1:
        img, bbox = detector.findFaces(img)
        # img, name, ct = faceRecognition(img)
        pError0, pError1 = trackFace(bbox, pid, pError0, pError1)

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
    window.after(10, update_frame)

update_frame()
window.mainloop()

# datn.streamoff()
# datn.land()
# video.release()
# cv2.destroyAllWindows()

