""" Depracated """

import numpy
import cv2
import time
import mss
from PIL import ImageGrab
import subprocess
import find

# title of our window
title = "Hacks"
# set start time to current time
start_time = time.time()
# displays the frame rate every 2 second
display_time = 2
# Set primarry FPS to 0
fps = 0
# Load mss library as sct
sct = mss.mss()
# Set monitor size to capture to MSS
monitor = {"top": 40, "left": 0, "width": 800, "height": 640}

types_of_obstacles = 4

def draw(img):
    # to ger real color we do this:
    dino_coords = []
    obstacle_coords = []

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    dino_coords = find.find_dino(img)

    for i in range(1, types_of_obstacles+1):
        obstacle_coords.append(find.find_obstacle(img, i))

    screen = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(screen, dino_coords[0][0], dino_coords[0][1], (0, 255, 0), 2)

    for coords in obstacle_coords:
        cv2.rectangle(screen, coords[0], coords[1], (0, 0, 255), 2)

    # subprocess.call("clear")
    # print(f'dino_coords: {dino_coords}, obstacle_coords: {obstacle_coords}')

    return screen

def main():
    global fps, start_time
    while True:
        # Get raw pixels from the screen, save it to a Numpy array
        img = numpy.array(sct.grab(monitor))

        screen = draw(img)
        cv2.imshow(title, screen)

        fps+=1
        TIME = time.time() - start_time
        if (TIME) >= display_time :
            print("FPS: ", fps / (TIME))
            fps = 0
            start_time = time.time()
        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
