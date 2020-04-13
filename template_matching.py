import cv2 as cv
import numpy as np

def get_score(img):
    values = []

    for i in range(10):
        exec(f"template{i} = cv.imread('ManualOCR/scores/number{i}.png',0)")

    for i in range(10):
        exec(("res = cv.matchTemplate(img,template{},cv.TM_CCOEFF_NORMED)\n" +
            "threshold = 0.95\n" +
            "loc = np.where( res >= threshold)\n" +
            "values.append(loc[1])").format(i))

    coords = []

    for v in values:
        for num in v:
            coords.append(num)

    coords.sort()
    digits = []

    for coord in coords:
        for c, v in enumerate(values):
            if coord in v:
                digits.append(c)
                break

    number = ''
    for d in digits:
        number += str(d)

    return int(number) if number != '' else ''
