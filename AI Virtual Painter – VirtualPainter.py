import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm
from flask import Flask,render_template,Response
app=Flask(__name__)


class AirCanvas():
    pass
    
def main():
    app=Flask(__name__)

    #######################
    brushThickness = 25
    eraserThickness = 100
    ########################


    folderPath = "Header"
    myList = os.listdir(folderPath)
    print(myList)
    overlayList = []
    for imPath in myList:
        image = cv2.imread(f'{folderPath}/{imPath}')
        overlayList.append(image)
    print(len(overlayList))
    header = overlayList[0]
    drawColor = (255, 0, 255)

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    detector = htm.handDetector(detectionCon=0.65,maxHands=1)
    xp, yp = 0, 0
    imgCanvas = np.zeros((720, 1280, 3), np.uint8)

    while True:

        # 1. Import image
        success, img = cap.read()
        img = cv2.flip(img, 1)

        # 2. Find Hand Landmarks
        img = detector.findHands(img)
        lmList, _ = detector.findPosition(img, draw=False)

        if len(lmList) != 0:

            # print(lmList)

            # tip of index and middle fingers
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]

            # 3. Check which fingers are up
            fingers = detector.fingersUp()
            # print(fingers)

            # 4. If Selection Mode - Two finger are up
            if fingers[1] and fingers[2]:
                # xp, yp = 0, 0
                print("Selection Mode")
                # # Checking for the click
                if y1 < 125:
                    if 250 < x1 < 450:
                        header = overlayList[0]
                        drawColor = (255, 0, 255)
                    elif 550 < x1 < 750:
                        header = overlayList[1]
                        drawColor = (255, 0, 0)
                    elif 800 < x1 < 950:
                        header = overlayList[2]
                        drawColor = (0, 255, 0)
                    elif 1050 < x1 < 1200:
                        header = overlayList[3]
                        drawColor = (0, 0, 0)
                cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

            # 5. If Drawing Mode - Index finger is up
            if fingers[1] and fingers[2] == False:
                cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
                print("Drawing Mode")
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)

                if drawColor == (0, 0, 0):
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
                
                else:
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

                xp, yp = x1, y1


            # Clear Canvas when all fingers are up
            if all (x >= 1 for x in fingers):
                imgCanvas = np.zeros((720, 1280, 3), np.uint8)

        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img,imgInv)
        img = cv2.bitwise_or(img,imgCanvas)


        # Setting the header image
        img[0:125, 0:1280] = header
        #img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
        # cv2.imshow("Image", img)
        # cv2.imshow("Canvas", imgCanvas)
        # cv2.imshow("Inv", imgInv)
        # cv2.waitKey(1)
        
        ret,buffer=cv2.imencode('.jpg',img)
        frame=buffer.tobytes()
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
@app.route('/')
def index():
    #return "HELLO"
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(main(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    # from werkzeug.serving import run_simple
    # run_simple('localhost', 9000, app)
    app.run(debug=True)
    
