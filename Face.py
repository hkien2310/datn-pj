import cv2
import mediapipe as mp
from djitellopy import tello
import math

# datn = tello.Tello()
# datn.connect()
# speed = 20

class FaceDetection():
    def __init__(self, minDetectionCon = 0.5):
        self.minDetectionCon = minDetectionCon

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection()

    def findFaces(self, img, draw=True):
        self.results = self.faceDetection.process(img)
        if self.results.detections:
            if draw:
                for id, detection in enumerate(self.results.detections):
                    #mpDraw.draw_detection(img,detection)
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, ic = img.shape
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    cv2.rectangle(img, bbox, (0,255,0), 2)
                    cv2.putText(img,f'{int(detection.score[0]*100)}%',(bbox[0],bbox[1]-20),
                                cv2.FONT_HERSHEY_PLAIN, 2,(0,255,0))
        return img

class FaceMesh():
    def __init__(self, staticimagemode = False, maxnumfaces=1, refinelandmarks=False, mindetectioncon=0.5, mintrackingcon=0.5):
        self.staticimagemode = staticimagemode
        self.maxnumfaces = maxnumfaces
        self.refinelandmarks= refinelandmarks
        self.mindetectioncon = mindetectioncon
        self.mintrackingcon = mintrackingcon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticimagemode, self.maxnumfaces, self.refinelandmarks,
                                                 self.mindetectioncon, self.mintrackingcon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self,img):
        self.results = self.faceMesh.process(img)
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS
                                           ,self.drawSpec, self.drawSpec)
                #for id, lm in enumerate(faceLms.landmarks):
        return img

class Hand():
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
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS,
                                               self.mpDraw.DrawingSpec(color=(255, 0, 0)),
                                               self.mpDraw.DrawingSpec(color=(255, 0, 0)))
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

    def ControlTello(self,lmList):
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
            #     datn.send_rc_control(0,0,0,0)
            # if totalFingers == 1:
            #     datn.send_rc_control(0,speed,0,0)
            # if totalFingers == 2:
            #     datn.send_rc_control(0,-speed,0,0)
            # if totalFingers == 3:
            #     datn.send_rc_control(speed,0,0,0)
            # if totalFingers == 4:
            #     datn.send_rc_control(-speed,0,0,0)


class Pose():
    def __init__(self,taticimagemode=False, modelcomplexity=1, smoothlandmarks=True,
                 enablesegmentation=False, smoothsegmentation=True, mindetectioncon=0.5, mintrackingcon=0.5):
        self.taticimagemode = taticimagemode
        self.modelcomplexity = modelcomplexity
        self.smoothlandmarks = smoothlandmarks
        self.enablesegmentation = enablesegmentation
        self.smoothsegmentation = smoothsegmentation
        self.mindetectioncon = mindetectioncon
        self.mintrackingcon = mintrackingcon

        self.mpPose = mp.solutions.pose
        self.PoseDetection = self.mpPose.Pose(self.taticimagemode, self.modelcomplexity, self.smoothlandmarks,
                                              self.enablesegmentation, self.smoothsegmentation,
                                              self.mindetectioncon, self.mintrackingcon)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img):
        self.results = self.PoseDetection.process(img)
        if self.results.pose_landmarks:
            # for poseLms in results.pose_landmarks:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS,
                                  self.mpDraw.DrawingSpec(color=(0, 0, 255)),
                                  self.mpDraw.DrawingSpec(color=(0, 0, 255)))
        return img

    def findPosition(self, img, draw=True):
        xList = []
        yList = []
        self.lmList = []
        bbox = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                # print(self.lmList)
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
            if draw:
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        return self.lmList, bbox

    def ControlDrone(self, img, p1, p2, p3, p4, p5, p6, p7, draw=True):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]
        x4, y4 = self.lmList[p4][1:]
        x5, y5 = self.lmList[p5][1:]
        x6, y6 = self.lmList[p6][1:]
        x7, y7 = self.lmList[p7][1:]

        cacu1 = math.degrees(math.atan2(y2-y1,x2-x1)-math.atan2(y2-y3,x2-x3))
        cacu2 = math.degrees(math.atan2(y5-y4,x5-x4)-math.atan2(y5-y6,x5-x6))
        if cacu1<0:
            cacu1 += 360
        if cacu2<0:
            cacu2 += 360

        if 250<cacu1<290:
            print("Left")

        if 80<cacu2<120:
            print("Right")

        if 80<cacu1<120:
            print("Backward")

        if 250<cacu2<290:
            print("Forward")

        if 170<cacu1<200:
            print("Down")

        if 150<cacu2<180:
            print("Up")
        # print("cacu1 =", cacu1)
        # print("cacu2 =", cacu2)

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





def main():
    cap = cv2.VideoCapture(0)
    detector = FaceDetection()
    detector1 = FaceMesh()
    detector2 = Hand()
    detector3 = Pose()
    while True:
        _, img = cap.read()
        # imgd = detector.findFaces(img)
        # imgm = detector1.findFaceMesh(img)
        # imgh = detector2.findHands(img)
        # lmList2, bbox2 = detector2.findPosition(imgh)
        # detector2.ControlTello(lmList2)
        imgp = detector3.findPose(img)
        lmList3, bbox3 = detector3.findPosition(imgp, False)
        if len(lmList3) != 0:
            detector3.ControlDrone(imgp, 15, 11, 23, 16, 12, 24, 0)
        # cv2.imshow('Output', imgd)
        # cv2.imshow('Output', imgm)
        # cv2.imshow('Output', imgh)
        cv2.imshow('Output', imgp)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

