#!/usr/bin/env python2

import time

import rospy
from geometry_msgs.msg import Twist
import std_msgs
import scipy.signal
import numpy as np
import cv2
from sensor_msgs.msg import Image
import cv_bridge

import util
import pedestrian

bridge = cv_bridge.CvBridge()
road_lines_pub = rospy.Publisher("/hippos/debug/road_lines", Image, queue_size=1)
drive = rospy.Publisher(util.topics.drive, Twist, queue_size=1)

pedestrian_flag = False

mode = "normal"


def at_crosswalk(frame):
    a = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB))[1]
    red = a > 200
    row_sums = red.sum(axis=0)
    peaks, _ = scipy.signal.find_peaks(row_sums)
    return red[-1].any() and len(peaks) > 1


def img_callback(ros_image):
    global mode
    frame = bridge.imgmsg_to_cv2(ros_image)
    h, s, v = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
    road_lines = np.logical_and(s < 50, v > 150)
    road_lines_pub.publish(bridge.cv2_to_imgmsg(road_lines.astype(np.uint8) * 255))

    move = Twist()
    move.angular.z = steer(road_lines)
    if mode == "normal":
        move.linear.x = 0.1
        if at_crosswalk(frame):
            mode = "wait_for_ped"
            move.linear.x = 0
        if move.angular.z != 0:
            move.linear.x = 0
    elif mode == "wait_for_ped":
        if pedestrian.is_pedestrian(frame):
            mode = "wait_for_no_ped"
    elif mode == "wait_for_no_ped":
        if not pedestrian.is_pedestrian(frame):
            move.linear.x = 0.2
            drive.publish(move)
            mode = "clear"
    elif mode == "clear":
        a = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB))[1] > 200
        if not a.any():
            mode = "normal"
        move.linear.x = 0.2

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


def set_warning(message):
    global pedestrian_flag
    pedestrian_flag = util.destringify(message)


def plate_message(location, plate_number):
    """Format a message for the license plate scorer"""
    return "Hippos,MitiFizz,{},{}".format(location, plate_number)


if __name__ == "__main__":
    rospy.init_node("controller", anonymous=True)
    plates = rospy.Publisher(util.topics.plates, std_msgs.msg.String, queue_size=1)
    warning = rospy.Subscriber(util.topics.warning, std_msgs.msg.String, set_warning)

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
    rospy.sleep(200)

    plates.publish(plate_message(-1, "AA00"))
