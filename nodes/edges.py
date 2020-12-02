#!/usr/bin/env python2

"""
Subscribe to license plate image topics and itentify them with a neural network.
"""

import time

import cv2
import numpy as np
import rospy
import scipy.ndimage
import sensor_msgs
import std_msgs
import cv_bridge

import util

bridge = cv_bridge.CvBridge()
fill_pub = rospy.Publisher("/hippos/debug/filled", sensor_msgs.msg.Image, queue_size=1)
edge_debug = rospy.Publisher("/hippos/debug/edges", sensor_msgs.msg.Image, queue_size=1)
edge_pub = rospy.Publisher(util.plate_topics.edges, std_msgs.msg.String, queue_size=1)


def light_grey(image, blur_sigma=0.5, s_max=20, v_min=87, v_max=240):
    _, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    if blur_sigma is not None:
        s = scipy.ndimage.gaussian_filter(s, blur_sigma)
        v = scipy.ndimage.gaussian_filter(v, blur_sigma)

    return np.logical_and(s < s_max, np.logical_and(v_min < v, v < v_max))


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


def filter_contours(contours, min_area=1000, min_solidity=0.7):
    return [
        c
        for c in contours
        if cv2.contourArea(c) > min_area
        and aspect_ratio(c) < 4
        and solidity(c) > min_solidity
    ]


def aspect_ratio(c):
    _, (w, h), _ = cv2.minAreaRect(c)
    return max(w, h) / min(w, h)


def solidity(c):
    return cv2.contourArea(c) / cv2.contourArea(cv2.convexHull(c))


def draw_contours_bw(image, contours):
    cv2.drawContours(image, contours, -1, 255, 1)


def grey(image, debug=False):
    bottom = image[300:]
    grey = light_grey(bottom)
    plates = fill_holes(grey)
    if debug:
        fill_pub.publish(bridge.cv2_to_imgmsg(plates.astype(np.uint8) * 255))

    contours = filter_contours(find_contours(plates))
    edges = np.zeros(bottom.shape[:2]).astype(np.uint8)
    draw_contours_bw(edges, contours)

    if debug:
        edge_debug.publish(bridge.cv2_to_imgmsg(edges.astype(np.uint8) * 255))

    edge_pub.publish(util.stringify((edges, bottom)))


def process_frame(ros_image):
    frame = bridge.imgmsg_to_cv2(ros_image, "bgr8")
    grey(frame, debug=True)


if __name__ == "__main__":
    rospy.init_node("edges", anonymous=True)
    rospy.Subscriber(util.topics.camera, sensor_msgs.msg.Image, process_frame)

    time.sleep(1)

    rospy.spin()
