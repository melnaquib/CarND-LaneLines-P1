# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

from moviepy.editor import VideoFileClip
from IPython.display import HTML

def draw_lines(img, lines, color=[0, 0, 255], thickness=2):
    if not lines:
        return
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    ignore_mask_color = 0

    gaussianBlur_kernel_size = 3

    filter_color_lower = np.array([180, 0, 0], dtype="uint16")
    filter_color_upper = np.array([255, 200, 150], dtype="uint16")

    canny_threshold_1 = 10
    canny_threshold_2 = 50

    hough_min_line_len = 40
    hough_max_line_gap = 40
    hough_rho = 1
    hough_theta = np.pi / 360
    hough_threshold = 5

    #    10 # distance resolution in pixels of the Hough grid
    # theta = np.pi/180 # angular resolution in radians of the Hough grid
    # threshold = 1     # minimum number of votes (intersections in Hough grid cell)
    # min_line_length = 200 #minimum number of pixels making up a line
    # max_line_gap = 25    # maximum gap in pixels between connectable line segments
    # line_image = np.copy(image)*0

    cvt = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
    smooth = cv2.GaussianBlur(cvt, (gaussianBlur_kernel_size, gaussianBlur_kernel_size), 0)
    gray = smooth[:, :, 0]
    mask = cv2.inRange(smooth, filter_color_lower, filter_color_upper)
    w = image.shape[1]
    h = int(image.shape[0] * 0.6)
    vertices = np.array([[[0, 0], [0, h], [w, h], [w, 0]]], np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    edges = cv2.Canny(mask, canny_threshold_1, canny_threshold_2)

    lines = cv2.HoughLinesP(edges, hough_rho, hough_theta, hough_threshold, np.array([]),
                            minLineLength=hough_min_line_len, maxLineGap=hough_max_line_gap)

    result = np.copy(image)
    draw_lines(result, lines)

    return result


x = 5

if x is not None:
    print ('DBG01')