import cv2
import mediapipe as mp
import numpy as np

class HandDetector:
    def __init__(self, staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.staticMode,
                                        max_num_hands=self.maxHands,
                                        model_complexity=modelComplexity,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True, flipType=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        return allHands, img

    def fingersUp(self, myHand):
        fingers = []
        myHandType = myHand["type"]
        myLmList = myHand["lmList"]
        if self.results.multi_hand_landmarks:

            # Thumb
            if myHandType == "Right":
                if myLmList[self.tipIds[0]][0] > myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if myLmList[self.tipIds[0]][0] < myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # 4 Fingers
            for id in range(1, 5):
                if myLmList[self.tipIds[id]][1] < myLmList[self.tipIds[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

def interpolate_points(p1, p2, num_points=20):
    x = np.linspace(p1[0], p2[0], num_points)
    y = np.linspace(p1[1], p2[1], num_points)
    return list(zip(map(int, x), map(int, y)))

def sketch():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

    canvas = None
    previous_position = None
    color = (0, 0, 0)
    eraser_mode = False
    background_white = False

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        if canvas is None:
            canvas = 255 * np.ones_like(img)

        hands, img = detector.findHands(img, draw=False, flipType=True)

        if hands:
            hand1 = hands[0]
            lmList1 = hand1["lmList"]
            fingers1 = detector.fingersUp(hand1)

            x, y = lmList1[8][0:2]

            if fingers1 == [0, 1, 0, 0, 0]:#1
                color = (0, 0, 0)
                eraser_mode = False
            elif fingers1 == [1, 1, 1, 1, 1]:#5
                save_path = "/content/drive/MyDrive/Last_Dance/Sketch2Image_Retrieval/sketch_data/drawing.png"
                cv2.imwrite(save_path, canvas)
                print(f"그림이 {save_path}에 저장되었습니다.")
                break
            elif fingers1 == [0, 1, 1, 0, 0]:#2
                color = (0, 255, 0)
                eraser_mode = False
            elif fingers1 == [0, 1, 1, 1, 0]:  # 3
                color = (0, 0, 255)
                eraser_mode = False
            elif fingers1 == [0, 1, 1, 1, 1]:  # 4
                eraser_mode = True

            if fingers1[1] == 1 and not background_white:
                if previous_position is not None:
                    points = interpolate_points(previous_position, (x, y))
                    for i in range(len(points) - 1):
                        if eraser_mode:
                            cv2.line(canvas, points[i], points[i + 1], (255, 255, 255), 50)
                        else:
                            cv2.line(canvas, points[i], points[i + 1], color, 5)
                previous_position = (x, y)
            else:
                previous_position = None

        display = img.copy()
        if background_white:
            display = 255 * np.ones_like(img)
        combined = cv2.addWeighted(display, 0.5, canvas, 0.5, 0)
        cv2.imshow("Live Feed with Drawing", combined)

        key = cv2.waitKey(1)
        if key == ord('r'):
            canvas = 255 * np.ones_like(img)
            break

    cap.release()
    cv2.destroyAllWindows()

