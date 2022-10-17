import numpy as np
import cv2 as cv


def find_circles(template_file):
    filled_file = template_file # FOR NOW USE SAME BLANK

    template = cv.imread(template_file)
    filled = cv.imread(filled_file)

    approx_bubble_ratio_to_page_width = 0.0092

    filled_h, filled_w, filled_channel = filled.shape
    h, w, channel = template.shape

    # aspect_ratio_h = filled_h / h
    # aspect_ratio_w = filled_w / w

    template = cv.resize(template, (5100, 6600))
    filled = cv.resize(filled, (5100, 6600))

    radius = round(approx_bubble_ratio_to_page_width * template.shape[1])
    radius_min = round(0.90 * radius)
    radius_max = round(1.10 * radius)

    # radius = int(radius_original_ACT * aspect_ratio_w)


    output = filled.copy()

    gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)

    # docstring of HoughCircles: HoughCircles
    # (image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 20,
                              param1=50, param2=30, minRadius=radius_min, maxRadius=radius_max)
    if circles is not None:
        detected_circles = np.uint16(np.around(circles))
        print(f"{len(detected_circles[0, :])} Circles found")

        for (x, y, r) in detected_circles[0, :]:
            cv.circle(output, (x, y), r, (0, 0, 255), 5)
    else:
        cv.circle(output, (1000, 1000), 40, (0, 255, 0), 20)

    cv.imshow('output', output)
    cv.waitKey(0)


templates = ['SAT_0.jpg', 'SAT_1.jpg',
             'PSAT_0.jpg', 'PSAT_1.jpg',
             'ACT_0.jpg', 'ACT_1.jpg',
             'PSAT_0_new.jpg', 'PSAT_1_new.jpg',]

for t in templates:
    find_circles(t)