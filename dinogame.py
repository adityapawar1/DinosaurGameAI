import numpy as np
import cv2
import time
import mss
from PIL import ImageGrab
import subprocess
import find
import pyautogui
from time import sleep

print('starting')

# title of our window
title = "Computer Vision"
# set start time to current time
start_time = time.time()
# displays the frame rate every 2 second
display_time = 2
# Set primarry FPS to 0
fps = 0
# Load mss library as sct
sct = mss.mss()
# Set monitor size to capture to MSS
high = 9999999999
monitor = {"top": 265, "left": 210, "width": 590, "height": 140}
update_dino = 1 # how many frames until you update the dino's postion
frame = 0
control_frame_interval = 3

cac_mask_upper = np.array([0, 0, 90])
cac_maks_lower = np.array([0, 0, 10])

contours = 0

ROI = [(43, 256), (136, 1300)]

time_pixel_buffer = 5

def findBoundingBoxesROI(contours):
    return_array = []
    for obstacle in contours:
        max_x = -1
        max_y = -1
        min_x = high
        min_y = high

        for coord in obstacle:
            coord = coord[0]

            max_x = coord[0] if coord[0] > max_x else max_x
            max_y = coord[1] if coord[1] > max_y else max_y
            min_x = coord[0] if coord[0] < min_x else min_x
            min_y = coord[1] if coord[1] < min_y else min_y

        return_array.append([(int(min_x), int(max_y)), (int(max_x), int(min_y))])

    return return_array

def findBoundingBoxesWithShift(contours):
    global ROI
    return_array = []
    for obstacle in contours:
        max_x = -1
        max_y = -1
        min_x = high
        min_y = high

        for coord in obstacle:
            coord = coord[0]

            max_x = coord[0] if coord[0] > max_x else max_x
            max_y = coord[1] if coord[1] > max_y else max_y
            min_x = coord[0] if coord[0] < min_x else min_x
            min_y = coord[1] if coord[1] < min_y else min_y

        return_array.append([(min_x + ROI[1][0], max_y + ROI[0][0]), (max_x + ROI[1][0], min_y + ROI[0][0])])

    return return_array

def drawBoundingBoxes(img, points, color):

    for coord in points:
        img = np.ascontiguousarray(img)
        cv2.rectangle(img, coord[0], coord[1], color, 2)

    return img

def findDistance(dino_coords, obstacles):
    dino_x = dino_coords[0][1][0]
    dino_mid_y = (dino_coords[0][0][1] + dino_coords[0][1][1]) / 2 # 226 when running

    cac_x = high
    closest = [[0, dino_mid_y], [0, dino_mid_y]]
    for coord in obstacles:
        if cac_x > coord[1][0]:
            cac_x = coord[0][0]
            closest = coord

    if cac_x == high:
        cac_x = dino_x

    cac_mid_y = (closest[0][1] + closest[1][1]) / 2
    cac_x -= time_pixel_buffer
    dist = cac_x - dino_x - time_pixel_buffer
    return dist, (int(dino_x), int(dino_mid_y)), (int(cac_x), int(cac_mid_y))

while True:
    # Get raw pixels from the screen, save it to a Numpy array
    try:
        img = np.array(sct.grab(monitor))
        obstacles = img[ROI[0][0]:ROI[0][1], ROI[1][0]:ROI[1][1]]
        # obstacles = img[43:256, 950:1300]
        gray = cv2.cvtColor(obstacles, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,90,255,cv2.THRESH_BINARY)

        out = obstacles.copy()

        edged = cv2.Canny(thresh, 30, 200)
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(out, contours, -1, (0, 0, 255), 2)
        points = findBoundingBoxesWithShift(contours)
        out = drawBoundingBoxes(img.copy(), points, (0, 0, 255))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if frame % update_dino == 0:
            dino_coords = find.find_dino(img)

        dist, start_point, end_point = findDistance(dino_coords, points)
        # print(dist)
        # print(start_point, end_point)

        if frame % control_frame_interval == 0:
            if dist > 20 and dist < 250:
                pyautogui.press('up')

        out = cv2.line(out, start_point, end_point, (0, 0, 0), 2)
        cv2.rectangle(out, dino_coords[0][0], dino_coords[0][1], (0, 255, 0), 2)

        # cv2.imshow(title, screen)
        cv2.imshow('processed', out)


        frame += 1

        if frame > 999999:
            frame = 0
        fps+=1
        TIME = time.time() - start_time
        if (TIME) >= display_time :
            print("FPS: ", fps / (TIME))
            fps = 0
            start_time = time.time()
        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"): # take only the lasy byte and compare
            cv2.destroyAllWindows()
            break
    except KeyboardInterrupt:
        print('program terminated')
        cv2.destroyAllWindows()
        break

subprocess.call('clear')
