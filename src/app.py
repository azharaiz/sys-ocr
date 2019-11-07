import os
import cv2 as cv
import imutils
import numpy as np
import pytesseract
from PIL import Image

location = os.getcwd() + "/img/car1.jpg"

image = cv.imread(location, cv.IMREAD_COLOR)
image = cv.resize(image, (640,480))

gray_scaled_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
smoothed_image = cv.bilateralFilter(gray_scaled_image, 11, 17, 17)
edged_image = cv.Canny(smoothed_image, 30, 200)

find_contour = cv.findContours(
    edged_image.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(find_contour)
sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]
screen_contours = None

for contour in sorted_contours:
    peri = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, 0.018 * peri, True)
    if len(approx) == 4:
        screen_contours = approx
        break

if screen_contours is None:
    detected = 0
    print ("No contour detected")
else:
    detected = 1

if detected == 1:
    cv.drawContours(image, [screen_contours], -1, (0, 255, 0), 3)

mask = np.zeros(smoothed_image.shape, np.uint8)
new_image = cv.drawContours(mask,[screen_contours],0,255,-1,)
new_image = cv.bitwise_and(image , image, mask=mask)

(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
cropped = gray_scaled_image[topx:bottomx+1, topy:bottomy+1]

cv.imshow('Image', cropped)
cv.waitKey(0)
cv.destroyAllWindows()
