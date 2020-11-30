#!/usr/bin/env python2

"""
Read license plates.
"""

import itertools as it
import os.path
import string

import cv2
import numpy as np
import scipy.ndimage

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


def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436

    From https://stackoverflow.com/a/46572063/3311667
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [x0, y0]


def partition(pred, iterable):
    """
    Use a predicate to partition entries into false entries and true entries

    partition(is_odd, range(10)) --> 0 2 4 6 8   and  1 3 5 7 9

    From https://docs.python.org/3/library/itertools.html#itertools-recipes
    """
    t1, t2 = it.tee(iterable)
    return it.ifilterfalse(pred, t1), filter(pred, t2)


def rects_from_lines(lines):
    horizontal, vertical = partition(lambda l: abs(l[0][1]) < np.pi / 6, lines)
    rects = []
    for (h1, h2), (v1, v2) in it.product(
        it.combinations(horizontal, 2), it.combinations(vertical, 2)
    ):
        if (h1 == h2).all() or (v1 == v2).all():
            continue

        corners = np.asarray(
            [
                intersection(h1, v1),
                intersection(h1, v2),
                intersection(h2, v2),
                intersection(h2, v1),
            ]
        )
        rects.append(sort_vertices(corners))

    return np.asarray(rects)


def sort_vertices(points):
    """From https://math.stackexchange.com/a/3651183/368176"""
    xc = points[..., 0]
    yr = points[..., 1]
    center_xc = np.sum(xc) / xc.shape
    center_yr = np.sum(yr) / yr.shape
    theta = np.arctan2(yr - center_yr, xc - center_xc)
    indices = np.argsort(theta)
    x = xc[indices]
    y = yr[indices]
    return np.asarray([x, y]).transpose()


def edge_score(rect, edges):
    """Compute the edge score of the rectangle: the percentage of the perimeter that lies along an edge."""
    rect_edges = np.zeros_like(edges)
    cv2.drawContours(rect_edges, [rect], 0, 255, 5)
    matching_edges = np.logical_and(edges, rect_edges)
    return np.count_nonzero(matching_edges) / cv2.arcLength(rect, closed=True)


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


letters = [
    cv2.split(
        cv2.cvtColor(
            util.imread(
                os.path.expanduser(
                    "~/ros_ws/src/hippos/nodes/templates/letter_{}.png".format(l)
                )
            ),
            cv2.COLOR_BGR2Lab,
        )
    )[0]
    for l in string.ascii_uppercase
]
numbers = [
    cv2.split(
        cv2.cvtColor(
            util.imread(
                os.path.expanduser(
                    "~/ros_ws/src/hippos/nodes/templates/number_{}.png".format(d)
                )
            ),
            cv2.COLOR_BGR2Lab,
        )
    )[2]
    for d in string.digits
]
locations = [
    cv2.split(
        cv2.cvtColor(
            util.imread(
                os.path.expanduser(
                    "~/ros_ws/src/hippos/nodes/templates/location_{}.png".format(n)
                )
            ),
            cv2.COLOR_BGR2Lab,
        )
    )[0]
    for n in string.digits[1:9]
]
original_shape = (1800, 600)

MATCH_THRESHOLD = 0.8


def class_matches(correlations, classes):
    predictions = []
    for i, corr in enumerate(correlations):
        if corr > MATCH_THRESHOLD:
            predictions.append(classes[i])
    return predictions


def predict(image):
    bottom = image[300:]
    grey = light_grey(bottom)
    plates = fill_holes(grey)

    contours = filter_contours(find_contours(plates))
    edges = np.zeros(bottom.shape[:2]).astype(np.uint8)
    draw_contours_bw(edges, contours)
    lines = cv2.HoughLines(edges, 1, np.pi / 30, 25)
    if lines is None:
        return []

    rects = rects_from_lines(lines)
    filtered = list(filter(lambda r: edge_score(r, edges) > 0.8, rects))

    predictions = []
    predicted_locs = []
    l, a, b = cv2.split(cv2.cvtColor(bottom, cv2.COLOR_BGR2Lab))
    for rect in filtered:
        warped_l = transform_to_rect(rect, l, original_shape)
        warped_b = transform_to_rect(rect, b, original_shape)

        location_correlation = []
        for location in locations:
            correlation = cv2.matchTemplate(
                warped_l[600:1200], location, cv2.TM_CCOEFF_NORMED
            )
            location_correlation.append(correlation.max())

        loc = class_matches(location_correlation, string.digits[1:9])
        if len(loc) != 1 or loc[0] in predicted_locs:
            continue

        letter1_correlation = []
        letter2_correlation = []
        for letter in letters:
            correlation = cv2.matchTemplate(
                warped_b[1200:, :300], letter, cv2.TM_CCOEFF_NORMED
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
                warped_b[1200:, 300:], number, cv2.TM_CCOEFF_NORMED
            )
            half = correlation.shape[1] // 2
            number1_correlation.append(correlation[:, :half].max())
            number2_correlation.append(correlation[:, half:].max())

        number1 = class_matches(number1_correlation, string.digits)
        number2 = class_matches(number2_correlation, string.digits)

        if len(letter1) == len(letter2) == len(number1) == len(number2) == 1:
            predictions.append(
                (loc[0], "".join((letter1[0], letter2[0], number1[0], number2[0])))
            )
            predicted_locs.append(loc[0])

    return predictions
