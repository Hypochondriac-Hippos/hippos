#!/usr/bin/env python2

"""
Read license plates.
"""

import os.path
import string

import cv2
import numpy as np

from rectangles import sort_vertices
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

already_read = set()

original_shape = (int(1800 * scale), int(600 * scale))

MATCH_THRESHOLD = 0.75


def transform_to_rect(rect, original, output_shape):
    output_points = sort_vertices(
        np.asarray(
            [
                [0, 0],
                [output_shape[1], 0],
                [output_shape[1], output_shape[0]],
                [0, output_shape[0]],
            ]
        )
    ).astype(np.float32)
    transform = cv2.getPerspectiveTransform(rect.astype(np.float32), output_points)
    return cv2.warpPerspective(
        original, transform, output_shape[1::-1], flags=cv2.INTER_CUBIC
    )


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


def predict(rect, bottom, debug=False):
    l, a, b = cv2.split(cv2.cvtColor(bottom, cv2.COLOR_BGR2Lab))
    warped = transform_to_rect(rect, l, original_shape)
    location_correlation = []
    for location in locations:
        correlation = cv2.matchTemplate(
            location_slice(warped),
            location,
            cv2.TM_CCOEFF_NORMED,
        )
        location_correlation.append(correlation.max())

    loc = class_matches(location_correlation, string.digits[1:9])
    if loc is None or loc in already_read:
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
        already_read.add(loc)
        return (loc, "".join((letter1, letter2, number1, number2)))
