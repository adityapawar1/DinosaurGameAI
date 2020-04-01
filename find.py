import numpy
import cv2

dino = cv2.imread('images/dino.png',0)
# c1 = cv2.imread('c1.png',0)
# c2 = cv2.imread('c2.png',0)
# c3 = cv2.imread('c3.png',0)
# c4 = cv2.imread('c4.png',0)
# uni = cv2.imread('uniobstacle.png',0)

dino_coords = []
obstacles_coords = []

def find_dino(screen):
    global dino
    w, h = dino.shape[::-1]
    # Apply template Matching
    res = cv2.matchTemplate(screen, dino, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    return [(top_left, bottom_right)]

# def find_obstacle(screen, num):
#     exec(f'global c{num}')
#
#     w, h = eval(f'c{num}.shape[::-1]')
#     # Apply template Matching
#     res = eval(f'cv2.matchTemplate(screen, c{num}, cv2.TM_CCOEFF)')
#
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#
#     top_left = max_loc
#     bottom_right = (top_left[0] + w, top_left[1] + h)
#     return (top_left, bottom_right)
