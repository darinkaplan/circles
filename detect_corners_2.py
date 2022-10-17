import numpy as np
import cv2
import imutils

# image = cv2.imread('sample_pg1.jpg')
image = cv2.imread('act_filled.jpg')

# convert the input image into
# grayscale color space
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 1)  # apply blur to roi
edged = cv2.Canny(blur, 10, 50)  # apply canny to roi

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gray = cv2.bilateralFilter(gray, 5, 5, 1)
# edged = cv2.Canny(gray, 10, 50)

cv2.imshow("gray", gray)
cv2.imshow("blur", blur)
cv2.imshow("edged", edged)
cv2.waitKey(0)



# find contours in the edged image, keep only the largest
# ones, and initialize our screen contour
cnts = cv2.findContours(gray.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
screenCnt = None

# loop over our contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.015 * peri, True)
    # if our approximated contour has four points, then
    # we can assume that we have found our screen
    if len(approx) == 4:
        screenCnt = approx
        break

cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 10)
cv2.imshow("Screen", image)
cv2.waitKey(0)



# Find my contours
contours = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

cnt = sorted(contours, key=cv2.contourArea, reverse=True)
