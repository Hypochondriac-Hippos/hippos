#!/usr/bin/env python2

import time

import rospy
from geometry_msgs.msg import Twist
import std_msgs
import numpy as np
import cv2
from sensor_msgs.msg import Image
import cv_bridge

import util

bridge = cv_bridge.CvBridge()
road_lines_pub = rospy.Publisher("/hippos/debug/road_lines", Image, queue_size=1)
drive = rospy.Publisher(util.topics.drive, Twist, queue_size=1)


def img_callback(ros_image):
    frame = bridge.imgmsg_to_cv2(ros_image)
    h, s, v = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
    road_lines = np.logical_and(s < 50, v > 150)
    road_lines_pub.publish(bridge.cv2_to_imgmsg(road_lines.astype(np.uint8) * 255))

    move = Twist()
    move.angular.z = steer(road_lines)
    move.linear.x = 0.1 if move.angular.z == 0 else 0
    drive.publish(move)


def steer(road_lines):
    if road_lines[-1, -1]:
        return 0

    magnitude = 1

    bottom_row_on = road_lines[-1].nonzero()[0]
    right_row_on = road_lines[:, -1].nonzero()[0]
    rightmost = bottomest = None
    if len(bottom_row_on) > 0:
        rightmost = road_lines.shape[1] - bottom_row_on.max()
    if len(right_row_on) > 0:
        bottomest = road_lines.shape[0] - right_row_on.max()

    if rightmost is None and bottomest is None:
        return 0
    if rightmost is None and bottomest is not None:
        return -magnitude
    if rightmost is not None and bottomest is None:
        return magnitude
    if rightmost < bottomest:
        return magnitude
    return -magnitude


def plate_message(location, plate_number):
    """Format a message for the license plate scorer"""
    return "Hippos,MitiFizz,{},{}".format(location, plate_number)


if __name__ == "__main__":
    rospy.init_node("controller", anonymous=True)
    plates = rospy.Publisher(util.topics.plates, std_msgs.msg.String, queue_size=1)
    warning = rospy.Subscriber(util.topics.warning, std_msgs.msg.String, queue_size=1)

    global ped_flag

    time.sleep(1)
    vel_msg = Twist()
    plates.publish(plate_message(0, "AA00"))

    # Brute force circuit
    vel_msg.linear.x = 0.272
    drive.publish(vel_msg)
    rospy.sleep(1.29)

    vel_msg.linear.x = 0
    vel_msg.angular.z = 1.1
    drive.publish(vel_msg)
    rospy.sleep(1.68)

    vel_msg.angular.z = 0
    drive.publish(vel_msg)

    camera = rospy.Subscriber(util.topics.camera, Image, img_callback)
    rospy.spin()

    plates.publish(plate_message(-1, "AA00"))
