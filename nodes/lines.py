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
fill_pub = rospy.Publisher("/hippos/debug/filled", sensor_msgs.msg.Image, queue_size=1)
edge_pub = rospy.Publisher(util.plate_topics.edges, sensor_msgs.msg.Image, queue_size=1)


def lines(edges, debug=False):
    lines = cv2.HoughLines(edges, 1, np.pi / 30, 20)
    if lines is None:
        return
    if debug:
        line_pub.publish(
            bridge.cv2_to_imgmsg(draw_lines(bottom, lines), encoding="bgr8")
        )


def process_frame(ros_image):
    frame = bridge.imgmsg_to_cv2(ros_image, "bgr8")
    grey(frame)


if __name__ == "__main__":
    rospy.init_node("grey", anonymous=True)
    score = rospy.Publisher(util.topics.plates, std_msgs.msg.String, queue_size=1)
    rospy.Subscriber(util.plate_topics.edges, sensor_msgs.msg.Image, process_frame)

    time.sleep(1)

    rospy.spin()
