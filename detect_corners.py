import numpy as np
import cv2

image = cv2.imread('sample_pg1.jpg')
# image = cv2.imread('act_filled.jpg')

# convert the input image into
# grayscale color space
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 1)  # apply blur to roi
canny = cv2.Canny(blur, 10, 50)  # apply canny to roi

# Find my contours
contours = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
cnt = sorted(contours, key=cv2.contourArea, reverse=True)
docCnt = cnt[0]
cv2.drawContours(image, cnt[:1], -1, (0, 255, 0), 20)

x, y, w, h = cv2.boundingRect(docCnt)
trimmed = image[y:y + h, x:x + w]

# the window showing output image with corners
cv2.imshow('Image with Borders', image)
cv2.imshow('trimmed', trimmed)

# De-allocate any associated memory usage
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
