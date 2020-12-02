#!/usr/bin/env python2

"""
Subscribe to license plate image topics and itentify them with a neural network.
"""

import time

import cv2
import numpy as np
import rospy
import sensor_msgs
import std_msgs
import cv_bridge

import util

bridge = cv_bridge.CvBridge()
line_debug = rospy.Publisher("/hippos/debug/lines", sensor_msgs.msg.Image, queue_size=1)
line_pub = rospy.Publisher(util.plate_topics.lines, std_msgs.msg.String, queue_size=1)


def draw_lines(image, lines):
    draw = image.copy()
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        if abs(theta) < np.pi / 6:
            colour = (255, 0, 0)
        else:
            colour = (0, 0, 255)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(draw, (x1, y1), (x2, y2), colour, 2)
    return draw


def lines(edges, bottom, debug=False):
    lines = cv2.HoughLines(edges, 1, np.pi / 30, 20)
    if lines is None:
        return
    if debug:
        line_debug.publish(
            bridge.cv2_to_imgmsg(draw_lines(bottom, lines), encoding="bgr8")
        )

    line_pub.publish(util.stringify((lines, edges, bottom)))


def process_frame(in_msg):
    edges, bottom = util.destringify(in_msg)
    lines(edges, bottom, debug=True)


if __name__ == "__main__":
    rospy.init_node("lines", anonymous=True)
    rospy.Subscriber(util.plate_topics.edges, std_msgs.msg.String, process_frame)
    time.sleep(1)

    rospy.spin()
