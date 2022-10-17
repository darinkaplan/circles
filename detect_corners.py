import numpy as np
import cv2

# image = cv2.imread('sample_pg1.jpg')
#image = cv2.imread('act_filled.jpg')
image = cv2.imread('sample_skewed.jpg')

# convert the input image into
# grayscale color space
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 1)  # apply blur to roi
canny = cv2.Canny(blur, 10, 50)  # apply canny to roi

# Find my contours
contours = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
cnt = sorted(contours, key=cv2.contourArea, reverse=True)
docCnt = cnt[0]

cv2.drawContours(image, cnt[:1], -1, (0, 255, 0), 5)
x, y, w, h = cv2.boundingRect(docCnt)
trimmed = image[y:y + h, x:x + w]

# NOW WORK ON WARPING TO REMOVE SKEW
rect = np.zeros((4, 2), dtype = "float32")
s = docCnt.sum(axis=2)

# the top-left point has the smallest sum whereas the
# bottom-right has the largest sum
rect[0] = docCnt[np.argmin(s)]
rect[2] = docCnt[np.argmax(s)]

# compute the difference between the points -- the top-right
# will have the minimum difference and the bottom-left will
# have the maximum difference
diff = np.diff(docCnt, axis = 2)
rect[1] = docCnt[np.argmin(diff)]
rect[3] = docCnt[np.argmax(diff)]

# now that we have our rectangle of points, let's compute
# the width of our new image
(tl, tr, br, bl) = rect
widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
# ...and now for the height of our new image
heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
# take the maximum of the width and height values to reach
# our final dimensions
maxWidth = max(int(widthA), int(widthB))
maxHeight = max(int(heightA), int(heightB))
# construct our destination points which will be used to
# map the screen to a top-down, "birds eye" view
dst = np.array([
	[0, 0],
	[maxWidth - 1, 0],
	[maxWidth - 1, maxHeight - 1],
	[0, maxHeight - 1]], dtype = "float32")
# calculate the perspective transform matrix and warp
# the perspective to grab the screen
M = cv2.getPerspectiveTransform(rect, dst)
warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))


# # List the output points in the same order as input
# # Top-left, top-right, bottom-right, bottom-left
# width, height, _ = image.shape
# dstPts = [[0, 0], [width, 0], [width, height], [0, height]]
# # Get the transform
# docCnt_32 = np.float32(docCnt)
# dstPts_32 = np.float32(dstPts)
#
# print(docCnt_32)
# print(dstPts_32)
#
# m = cv2.getPerspectiveTransform(docCnt_32, dstPts_32)
# # Transform the image
# final = cv2.warpPerspective(trimmed, m, (int(width), int(height)))


# the window showing output image with corners
cv2.imshow('Image with Borders', image)
cv2.imshow('trimmed', trimmed)
cv2.imshow('warp', warp)

# De-allocate any associated memory usage
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

