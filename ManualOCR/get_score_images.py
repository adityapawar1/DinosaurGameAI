import cv2
import numpy as np
import mss # for getting screenshots efficiently
import time

sct = mss.mss()

monitor = {"top": 265, "left": 210, "width": 590, "height": 140}
score_ROI = [(0, 80), (1005, 1200)]

name = 'scores/score{}.png'


for i in range(30):
    img = np.array(sct.grab(monitor))

    score_img = img[score_ROI[0][0]:score_ROI[0][1], score_ROI[1][0]:score_ROI[1][1]]
    gray_score = cv2.cvtColor(score_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(score_img,90,255,cv2.THRESH_BINARY)

    cv2.imwrite(name.format(i), thresh)

    time.sleep(0.3)
