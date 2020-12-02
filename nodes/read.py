#!/usr/bin/env python2

"""
Read license plates.
"""

import os.path
import random
import string

import cv2
import cv_bridge
import numpy as np
import rospy
import sensor_msgs
import scipy.ndimage

import rectangles
import util


def preprocessing(image):
    return cv2.split(
        cv2.cvtColor(
            cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2Lab,
        )
    )[0]


scale = 0.5

letters = [
    preprocessing(
        util.imread(
            os.path.expanduser(
                "~/ros_ws/src/hippos/nodes/templates/letter_{}.png".format(l)
            )
        )
    )
    for l in string.ascii_uppercase
]
numbers = [
    preprocessing(
        util.imread(
            os.path.expanduser(
                "~/ros_ws/src/hippos/nodes/templates/number_{}.png".format(d)
            )
        )
    )
    for d in string.digits
]
locations = [
    preprocessing(
        util.imread(
            os.path.expanduser(
                "~/ros_ws/src/hippos/nodes/templates/location_{}.png".format(n)
            )
        )
    )
    for n in string.digits[1:9]
]

original_shape = (int(1800 * scale), int(600 * scale))

MATCH_THRESHOLD = 0.8


def class_matches(correlations, classes):
    i = np.argmax(correlations)
    if correlations[i] > MATCH_THRESHOLD:
        return classes[i]


def location_slice(image):
    return image[image.shape[0] / 3 : image.shape[0] * 2 / 3, image.shape[1] / 2 :]


def letter_slice(image):
    return image[image.shape[0] * 2 / 3 :, : image.shape[1] / 2]


def number_slice(image):
    return image[image.shape[0] * 2 / 3 :, image.shape[1] / 2 :]


def draw_rects(image, rects):
    draw = image.copy()
    for r in rects:
        perturbation = [random.randint(-3, 3), random.randint(-3, 3)]  # Reduce overlap
        draw = cv2.polylines(draw, [r + perturbation], True, random_colour(), 2)
    return draw


def random_colour():
    h = random.randint(0, 255)
    s = random.randint(127, 255)
    v = random.randint(127, 255)
    point = np.uint8([[[h, s, v]]])
    return tuple(int(c) for c in cv2.cvtColor(point, cv2.COLOR_HSV2BGR)[0, 0])


rect_pub = rospy.Publisher("/hippos/debug/rects", sensor_msgs.msg.Image, queue_size=1)
bridge = cv_bridge.CvBridge()


def predict(lines, edges, bottom, debug=False):
    rects = rectangles.rects_from_lines(lines)
    rect = rectangles.take_max_edge_score(rects, edges)
    if rect is None:
        return
    if debug:
        rect_pub.publish(
            bridge.cv2_to_imgmsg(draw_rects(bottom, [rect]), encoding="bgr8")
        )

    l, a, b = cv2.split(cv2.cvtColor(bottom, cv2.COLOR_BGR2Lab))
    warped = rectangles.transform_to_rect(rect, l, original_shape)
    location_correlation = []
    for location in locations:
        correlation = cv2.matchTemplate(
            location_slice(warped),
            location,
            cv2.TM_CCOEFF_NORMED,
        )
        location_correlation.append(correlation.max())

    loc = class_matches(location_correlation, string.digits[1:9])
    if loc is None:
        return

    number1_correlation = []
    number2_correlation = []
    for number in numbers:
        correlation = cv2.matchTemplate(
            number_slice(warped), number, cv2.TM_CCOEFF_NORMED
        )
        half = correlation.shape[1] // 2
        number1_correlation.append(correlation[:, :half].max())
        number2_correlation.append(correlation[:, half:].max())

    number1 = class_matches(number1_correlation, string.digits)
    number2 = class_matches(number2_correlation, string.digits)

    if number1 is None or number2 is None:
        return

    letter1_correlation = []
    letter2_correlation = []
    for letter in letters:
        correlation = cv2.matchTemplate(
            letter_slice(warped), letter, cv2.TM_CCOEFF_NORMED
        )
        half = correlation.shape[1] // 2
        letter1_correlation.append(correlation[:, :half].max())
        letter2_correlation.append(correlation[:, half:].max())

    letter1 = class_matches(letter1_correlation, string.ascii_uppercase)
    letter2 = class_matches(letter2_correlation, string.ascii_uppercase)

    if letter1 is not None and letter2 is not None:
        return (loc, "".join((letter1, letter2, number1, number2)))
