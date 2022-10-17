import numpy as np
import cv2
import math

def get_approx_contour(contour, tol=.01):
    """Gets rid of 'useless' points in the contour."""
    epsilon = tol * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)

def calculate_contour_features(contour):
    """Calculates interesting properties (features) of a contour.

    We use these features to match shapes (contours). In this script,
    we are interested in finding shapes in our input image that look like
    a corner. We do that by calculating the features for many contours
    in the input image and comparing these to the features of the corner
    contour. By design, we know exactly what the features of the real corner
    contour look like - check out the calculate_corner_features function.

    It is crucial for these features to be invariant both to scale and rotation.
    In other words, we know that a corner is a corner regardless of its size
    or rotation. In the past, this script implemented its own features, but
    OpenCV offers much more robust scale and rotational invariant features
    out of the box - the Hu moments.
    """
    moments = cv2.moments(contour)
    huMoments = cv2.HuMoments(moments)
    # Log scale hu moments
    # for i in range(0, 7):
    #     huMoments[i] = -1 * math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i]))
    print(f"{huMoments}\n================")

    return huMoments


def calculate_corner_features():
    """Calculates the array of features for the corner contour.
    In practice, this can be pre-calculated, as the corners are constant
    and independent from the inputs.

    We load the img/corner.png file, which contains a single corner, so we
    can reliably extract its features. We will use these features to look for
    contours in our input image that look like a corner.
    """
    corner_img = cv2.imread('images/corner.png')
    corner_img_gray = cv2.cvtColor(corner_img, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(
        corner_img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # We expect to see only two contours:
    # - The "outer" contour, which wraps the whole image, at hierarchy level 0
    # - The corner contour, which we are looking for, at hierarchy level 1
    # If in trouble, one way to check what's happening is to draw the found contours
    # with cv2.drawContours(corner_img, contours, -1, (255, 0, 0)) and try and find
    # the correct corner contour by drawing one contour at a time. Ideally, this
    # would not be done at runtime.
    if len(contours) != 2:
        raise RuntimeError(
            'Did not find the expected contours when looking for the corner')

    # Following our assumptions as stated above, we take the contour that has a parent
    # contour (that is, it is _not_ the outer contour) to be the corner contour.
    # If in trouble, verify that this contour is the corner contour with
    # cv2.drawContours(corner_img, [corner_contour], -1, (255, 0, 0))
    corner_contour = next(ct
                          for i, ct in enumerate(contours)
                          if hierarchy[0][i][3] != -1)

    # Return the HuMoments of the image
    return calculate_contour_features(corner_contour)



def get_corners(contours):
    """Returns the 4 contours that look like a corner the most.

    In the real world, we cannot assume that the corners will always be present,
    and we likely need to decide how good is good enough for contour to
    look like a corner.
    This is essentially a classification problem. A good approach would be
    to train a statistical classifier model and apply it here. In our little
    exercise, we assume the corners are necessarily there."""
    corner_features = calculate_corner_features()
    return sorted(
        contours,
        key=lambda c: features_distance(
                corner_features,
                calculate_contour_features(c)))[:4]


def features_distance(f1, f2):
    return np.linalg.norm(np.array(f1) - np.array(f2))


image = cv2.imread('images/ACT_1.jpg')
# image = cv2.imread('sample_pg1.jpg')
# image = cv2.imread('act_filled.jpg')
# image = cv2.imread('images/sample_skewed.jpg')

# convert the input image into
# grayscale color space
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 1)  # apply blur to roi
canny = cv2.Canny(blur, 10, 50)  # apply canny to roi

# Find my contours
contours, hierarchy = cv2.findContours(
        canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
my_contours = map(get_approx_contour, contours)
cnt = sorted(contours, key=cv2.contourArea, reverse=True)
docCnt = cnt[0]


corners = get_corners(my_contours)
cv2.drawContours(image, corners, -1, (0, 255, 0), 3)
cv2.imshow('Corner image', image)
cv2.waitKey(0)





# for i in cnt:
# 	cv2.drawContours(image,i, -1, (0, 255, 0), 5)
# 	cv2.imshow('Image   with Borders', image)
# 	cv2.waitKey(0)

cv2.drawContours(image, cnt[:8], -1, (0, 255, 0), 5)
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
# cv2.imshow('warp', warp)

# De-allocate any associated memory usage
if cv2.waitKey(0):
    cv2.destroyAllWindows()

