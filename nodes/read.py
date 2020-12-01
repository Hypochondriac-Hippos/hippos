#!/usr/bin/env python2

"""
Read license plates.
"""

import os.path
import string

import cv2
import numpy as np
import scipy.ndimage

import rectangles
import util


def light_grey(image, blur_sigma=0.5, s_max=20, v_min=84):
    _, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    if blur_sigma is not None:
        s = scipy.ndimage.gaussian_filter(s, blur_sigma)
        v = scipy.ndimage.gaussian_filter(v, blur_sigma)

    return np.logical_and(s < s_max, v > v_min)


def fill_holes(image, dilation=2):
    if dilation is not None:
        v_struct = np.zeros((3, 3))
        v_struct[:, 1] = 1
        dilated = scipy.ndimage.binary_dilation(image, v_struct, iterations=dilation)
        filled = scipy.ndimage.binary_fill_holes(dilated)
        out = scipy.ndimage.binary_erosion(filled, v_struct, iterations=dilation + 1)
    else:
        out = scipy.ndimage.binary_fill_holes(image)

    return out


def find_contours(binary):
    return cv2.findContours(
        binary.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )[1]


def filter_contours(contours, min_area=500, min_solidity=0.7):
    return [
        c
        for c in contours
        if cv2.contourArea(c) > min_area and solidity(c) > min_solidity
    ]


def solidity(c):
    return cv2.contourArea(c) / cv2.contourArea(cv2.convexHull(c))


def draw_contours_bw(image, contours):
    cv2.drawContours(image, contours, -1, 255, 1)


def preprocessing(image):
    return cv2.split(
        cv2.cvtColor(
            cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2Lab,
        )
    )


scale = 0.5

letters = [
    preprocessing(
        util.imread(
            os.path.expanduser(
                "~/ros_ws/src/hippos/nodes/templates/letter_{}.png".format(l)
            )
        )
    )[2]
    for l in string.ascii_uppercase
]
numbers = [
    preprocessing(
        util.imread(
            os.path.expanduser(
                "~/ros_ws/src/hippos/nodes/templates/number_{}.png".format(d)
            )
        )
    )[2]
    for d in string.digits
]
locations = [
    preprocessing(
        util.imread(
            os.path.expanduser(
                "~/ros_ws/src/hippos/nodes/templates/location_{}.png".format(n)
            )
        )
    )[0]
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


def predict(image):
    bottom = image[300:]
    grey = light_grey(bottom)
    plates = fill_holes(grey)

    contours = filter_contours(find_contours(plates))
    edges = np.zeros(bottom.shape[:2]).astype(np.uint8)
    draw_contours_bw(edges, contours)
    lines = cv2.HoughLines(edges, 1, np.pi / 30, 20)
    if lines is None:
        return []

    rects = rectangles.rects_from_lines(lines)
    filtered = list(filter(lambda r: rectangles.edge_score(r, edges) > 0.8, rects))
    predictions = []
    predicted_locs = []
    l, a, b = cv2.split(cv2.cvtColor(bottom, cv2.COLOR_BGR2Lab))
    for rect in filtered:
        warped_l = rectangles.transform_to_rect(rect, l, original_shape)
        warped_b = rectangles.transform_to_rect(rect, b, original_shape)

        location_correlation = []
        for location in locations:
            correlation = cv2.matchTemplate(
                location_slice(warped_l),
                location,
                cv2.TM_CCOEFF_NORMED,
            )
            location_correlation.append(correlation.max())

        loc = class_matches(location_correlation, string.digits[1:9])
        if loc is None:
            continue

        letter1_correlation = []
        letter2_correlation = []
        for letter in letters:
            correlation = cv2.matchTemplate(
                letter_slice(warped_b), letter, cv2.TM_CCOEFF_NORMED
            )
            half = correlation.shape[1] // 2
            letter1_correlation.append(correlation[:, :half].max())
            letter2_correlation.append(correlation[:, half:].max())

        letter1 = class_matches(letter1_correlation, string.ascii_uppercase)
        letter2 = class_matches(letter2_correlation, string.ascii_uppercase)

        number1_correlation = []
        number2_correlation = []
        for number in numbers:
            correlation = cv2.matchTemplate(
                number_slice(warped_b), number, cv2.TM_CCOEFF_NORMED
            )
            half = correlation.shape[1] // 2
            number1_correlation.append(correlation[:, :half].max())
            number2_correlation.append(correlation[:, half:].max())

        number1 = class_matches(number1_correlation, string.digits)
        number2 = class_matches(number2_correlation, string.digits)

        if (
            letter1 is not None
            and letter2 is not None
            and number1 is not None
            and number2 is not None
        ):
            predictions.append((loc, "".join((letter1, letter2, number1, number2))))
            predicted_locs.append(loc[0])

    return predictions
