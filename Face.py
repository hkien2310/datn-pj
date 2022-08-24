import cv2
import mediapipe as mp


#Chương trình phát hiện khuôn mặt
class FaceDetection():
    def __init__(self, minDetectionCon = 0.5):
        self.minDetectionCon = minDetectionCon #Khai báo thông số của face_detection

        #Khai báo các biến
        self.mpFaceDetection = mp.solutions.face_detection # Tìm mặt
        self.mpDraw = mp.solutions.drawing_utils #Vẽ
        self.faceDetection = self.mpFaceDetection.FaceDetection()# Tìm mặt

    def findFaces(self, img, draw=True): # Tìm mặt
        bbox = [0, 0, 0, 0]
        #Kết quả trả về sau khi tìm mặt
        self.results = self.faceDetection.process(img)
        if self.results.detections:
            if draw:
                for id, detection in enumerate(self.results.detections):
                    #mpDraw.draw_detection(img,detection)
                    #Các giá trị của hộp giới hạn tương đối
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, ic = img.shape
                    #Các giá trị của hộp giới hạn
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    cv2.rectangle(img, bbox, (0,255,0), 2)
                    # print(bbox)
                    cv2.putText(img,f'{int(detection.score[0]*100)}%',(bbox[0],bbox[1]-20),
                                cv2.FONT_HERSHEY_PLAIN, 2,(0,255,0))
        return img, bbox

#Chương trình phát hiện và vẽ lưới trên khuôn mặt
class FaceMesh():
    def __init__(self, staticimagemode = False, maxnumfaces=1, refinelandmarks=False, mindetectioncon=0.5, mintrackingcon=0.5):
        #Khai báo các giá trị của thư viện face_mesh
        self.staticimagemode = staticimagemode
        self.maxnumfaces = maxnumfaces
        self.refinelandmarks= refinelandmarks
        self.mindetectioncon = mindetectioncon
        self.mintrackingcon = mintrackingcon

        #Khai báo các biến
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticimagemode, self.maxnumfaces, self.refinelandmarks,
                                                 self.mindetectioncon, self.mintrackingcon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self,img): # Tìm và vẽ lưới mặt
        self.results = self.faceMesh.process(img) #Kết quả
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS
                                           ,self.drawSpec, self.drawSpec)
                #for id, lm in enumerate(faceLms.landmarks):
        return img

#Chương trình phát hiện bàn tay
class Hand():
    def __init__(self,mode=False, maxHands=2, model_complexity = 1, detectionCon=0.5, trackCon=0.5 ):
        #Khai báo các giá trị của thư viện hands
        self.mode = mode
        self.maxHands = maxHands
        self.model_complexity = model_complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        #Khai báo các biến
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.model_complexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True): # Tìm và vẽ bàn tay
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

    def findPosition(self, img, handNo=0, draw=True): #Tìm và vẽ các điểm trên bàn tay
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


#Chương trình phát hiện dáng người
class Pose():
    def __init__(self,taticimagemode=False, modelcomplexity=1, smoothlandmarks=True,
                 enablesegmentation=False, smoothsegmentation=True, mindetectioncon=0.5, mintrackingcon=0.5):
        # Khai báo các giá trị của thư viện pose
        self.taticimagemode = taticimagemode
        self.modelcomplexity = modelcomplexity
        self.smoothlandmarks = smoothlandmarks
        self.enablesegmentation = enablesegmentation
        self.smoothsegmentation = smoothsegmentation
        self.mindetectioncon = mindetectioncon
        self.mintrackingcon = mintrackingcon

        #Khai báo các biến
        self.mpPose = mp.solutions.pose
        self.PoseDetection = self.mpPose.Pose(self.taticimagemode, self.modelcomplexity, self.smoothlandmarks,
                                              self.enablesegmentation, self.smoothsegmentation,
                                              self.mindetectioncon, self.mintrackingcon)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img): # Tìm và vẽ dáng người
        self.results = self.PoseDetection.process(img)
        if self.results.pose_landmarks:
            # for poseLms in results.pose_landmarks:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS,
                                  self.mpDraw.DrawingSpec(color=(0, 0, 255)),
                                  self.mpDraw.DrawingSpec(color=(0, 0, 255)))
        return img

    def findPosition(self, img, draw=True): #Tìm và vẽ các điểm trên dáng người
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

    cap.release() # Giải phóng camera
    cv2.destroyAllWindows()#Tắt hết cửa sổ


if __name__ == '__main__':
    main()

