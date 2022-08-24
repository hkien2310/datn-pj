from tkinter import *
from tkinter.ttk import *
import tkinter
import cv2
import PIL.Image,PIL.ImageTk
import mediapipe as mp
import Face


window = Tk()
window.title("Giao diện")
window.geometry("800x600")

imgbg = PhotoImage(file="background.jpg")
imgbg = cv2.resize(imgbg, (800,600))
imglogo = PhotoImage(file = 'logohaui.png')
imglogo = cv2.resize(imglogo, (50,50))


bt1, bt2, bt3 =0, 0, 0
def BaW():
    global bt1
    bt1 = 1 - bt1

def Blur():
    global bt2
    bt2 = 1 - bt2

def Findtheedge():
    global bt3
    bt3 = 1 - bt3

video = cv2.VideoCapture(0)
lbl = Label(window, text="Camera", font=("Arial",20))
lbl.place(x=250, y=0)
canvas = Canvas(window, width= 500, height=400, bg='red')
canvas.place(x=150, y=30)
bg = Label(window, image=imgbg)
bg.place(x=0,y=0)
button1 = Button(window, text='findFace', command=BaW)
button1.place(x=150, y=433)
button2 = Button(window, text='faceMesh', command=Blur)
button2.place(x=250, y=433)
button3 = Button(window, text='findHands', command=Findtheedge)
button3.place(x=350, y=433)

detector = Face.FaceDetection()
detector1 = Face.FaceMesh()
detector2 = Face.Hand()
detector3 = Face.Pose()

def update_frame():
    global canvas,photo
    ret, img = video.read()
    img = cv2.resize(img, dsize=None, fx=0.8, fy=0.85)
    # imgd = detector.findFaces(img)
    # imgm = detector1.findFaceMesh(img)
    # imgh = detector2.findHands(img)
    # lmList2, bbox2 = detector2.findPosition(imgh)
    # detector2.ControlTello(lmList2)
    # imgp = detector3.findPose(img)
    # lmList3, bbox3 = detector3.findPosition(imgp)
    if bt1==1:
        img = detector.findFaces(img)
    if bt2==1:
        img = detector1.findFaceMesh(img)
    if bt3==1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = detector2.findHands(img)
        lmList2, bbox2 = detector2.findPosition(img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0,0, image = photo, anchor=tkinter.NW)
    window.after(15, update_frame)

update_frame()
window.mainloop()