import cv2
import mediapipe as mp
from djitellopy import tello

class handDetector():
    def __init__(self,mode=False, maxHands=2, model_complexity = 1, detectionCon=0.5, trackCon=0.5 ):
        self.mode = mode
        self.maxHands = maxHands
        self.model_complexity = model_complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.model_complexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        # Chuyển sang hệ màu RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        self.lmList = []
        bbox = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)

        return self.lmList, bbox



def main():
    video = cv2.VideoCapture(0)
    #Gọi hàn tìm bàn tay
    detector = handDetector()
    Ids = [4, 8, 12, 16, 20]
    speed = 20
    while True:
        # Đọc ảnh từ camera drone
        #cap = datn.get_frame_read().frame
        _, cap = video.read()
        #Resize ảnh về kích thước 500x400
        cap = cv2.resize(cap, (500, 400))
        #Tìm bàn tay trong ảnh
        cap = detector.findHands(cap)
        #Tìm các điểm trên tay
        lmList, bbox =detector.findPosition(cap)
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
            """"
            if totalFingers == 0:
                datn.send_rc_control(0,0,0,0)
            if totalFingers == 1:
                datn.send_rc_control(0,speed,0,0)
            if totalFingers == 2:
                datn.send_rc_control(0,-speed,0,0)
            if totalFingers == 3:
                datn.send_rc_control(speed,0,0,0)
            if totalFingers == 4:
                datn.send_rc_control(-speed,0,0,0)
                """


        cv2.imshow('Display', cap)
        cv2.waitKey(1)



if __name__ == '__main__':
    """"
    datn = tello.Tello()
    # Kết nối với drone
    datn.connect()
    # Drone cất cánh
    datn.takeoff()
    #Bật camera
    datn.streamon()
    # In ra giá trị của pin
    print(datn.get_battery())
    main()
    datn.streamoff()
    datn.land()
"""
    main()
