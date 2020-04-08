import cv2
import numpy as np
import template_matching as ocr
import mss
import time
import pytesseract


file = 'scores/score{}.png'

sct = mss.mss()
# Set monitor size to capture to MSS
monitor = {"top": 265, "left": 210, "width": 590, "height": 140}

img = np.array(sct.grab(monitor))

score_ROI = [(0, 80), (1005, 1200)]

iter = 0
corr = 0
try:
    while True:

        img = np.array(sct.grab(monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        score_img = img[score_ROI[0][0]:score_ROI[0][1], score_ROI[1][0]:score_ROI[1][1]]
        ret, thresh = cv2.threshold(score_img,90,255,cv2.THRESH_BINARY)
        score = ocr.get_score(thresh)
        tess_score = pytesseract.image_to_string(score_img, config='digits')


        try:
            tess_score = int(tess_score)
        except:
            pass

        if score != '':
            iter += 1
            if score == tess_score:
                corr += 1
        else:
            corr += 1
            iter += 1


        print(score, tess_score)
        time.sleep(0.2)
except:
    print()
    print(f'Tess Accuracy: {corr/iter}')
