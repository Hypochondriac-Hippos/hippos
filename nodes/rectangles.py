#!/usr/bin/env python2

"""
Rectangle functions
"""

import itertools as it

import cv2
import numpy as np


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
        yield sort_vertices(corners)


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
    arc = cv2.arcLength(rect, closed=True)
    if arc < 1:
        return 0
    return np.count_nonzero(matching_edges) / arc


def take_max_edge_score(rects, edges):
    best = None
    maximum = None
    for r in rects:
        score = edge_score(r, edges)
        if maximum is None or score > maximum:
            best = r
            maximum = score

    return best


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
