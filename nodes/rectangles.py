#!/usr/bin/env python2

"""
Rectangle functions
"""

import itertools as it
import random
import time

import cv2
import cv_bridge
import numpy as np
import rospy
import sensor_msgs
import std_msgs

import util

rect_debug = rospy.Publisher("/hippos/debug/rects", sensor_msgs.msg.Image, queue_size=1)
rect_pub = rospy.Publisher(util.plate_topics.rects, std_msgs.msg.String, queue_size=1)


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


bridge = cv_bridge.CvBridge()


def find_best_rect(lines, edges, bottom, debug=False):
    rects = rects_from_lines(lines)
    rect = take_max_edge_score(rects, edges)
    if rect is None:
        return
    if debug:
        rect_debug.publish(
            bridge.cv2_to_imgmsg(draw_rects(bottom, [rect]), encoding="bgr8")
        )

    rect_pub.publish(util.stringify((rect, bottom)))


def process_frame(in_msg):
    lines, edges, bottoms = util.destringify(in_msg)
    find_best_rect(lines, edges, bottoms, debug=True)


if __name__ == "__main__":
    rospy.init_node("rects", anonymous=True)
    rospy.Subscriber(util.plate_topics.lines, std_msgs.msg.String, process_frame)
    time.sleep(1)

    rospy.spin()
