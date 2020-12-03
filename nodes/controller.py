#!/usr/bin/env python2

from collections import namedtuple
import time

import rospy
from geometry_msgs.msg import Twist
import std_msgs
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

topics = namedtuple("Topics", ("camera", "clock", "drive", "plates", "warning"))(
    "/R1/pi_camera/image_raw", "/clock", "/R1/cmd_vel", "/license_plate", "/hippos/pedestrian"
)

nav_vals = {"Set Points": (329, 261), "Prev_vals": (0, 0)}

ped_flag = False

bridge = CvBridge()


def img_callback(data):
    global nav_vals
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
        smoothed = cv2.GaussianBlur(cv_image, (5, 5), 0)
        edges = cv2.Canny(smoothed, 40, 80)
        cropped_edges = edges[:, 850:]
        x1 = np.argmax(cropped_edges[719])
        x2 = np.argmax(cropped_edges[672])

        nav_vals["Prev_vals"] = (x1, x2)
    except CvBridgeError as e:
        print(e)


def steer():
    global nav_vals
    dx1 = nav_vals["Set Points"][0]-nav_vals["Prev_vals"][0]
    dx2 = nav_vals["Set Points"][1]-nav_vals["Prev_vals"][1]
    if dx1 > 20:
        if dx2 > 20:
            z = -0.1
        elif dx2 < -20:
            z = 0.5
        else:
            z = 0.2
    elif dx1 < -20:
        if dx2 > 20:
            z = -0.5
        elif dx2 < -20:
            z = 0.5
        else:
            z = -0.2
    else:
        z = 0

    return z


def ped_warning(data):
    global ped_flag
    if data is "True":
        ped_flag = True
    else:
        ped_flag = False



def plate_message(location, plate_number):
    """Format a message for the license plate scorer"""
    return "Hippos,MitiFizz,{},{}".format(location, plate_number)


if __name__ == "__main__":
    rospy.init_node("controller", anonymous=True)
    plates = rospy.Publisher(topics.plates, std_msgs.msg.String, queue_size=1)
    drive = rospy.Publisher(topics.drive, Twist, queue_size=1)
    camera = rospy.Subscriber(topics.camera, Image, img_callback)
    warning = rospy.Subscriber(topics.warning, std_msgs.msg.String, queue_size=1)

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

    # vel_msg.linear.x = 0.1
    # drive.publish(vel_msg)
    # rospy.sleep(0.2)
    # vel_msg.angular.z = 0
    # drive.publish(vel_msg)
    # rospy.sleep(11.1)

    while not rospy.is_shutdown():
        # if ped_flag:
        #     vel_msg.linear.x = 0
        #     vel_msg.angular.z = 0
        #     drive.publish(vel_msg)
        #     rospy.sleep(0.1)
        # else:
        #     vel_msg.linear.x = 0.1
        #     vel_msg.angular.z = steer()
            print(str(nav_vals))
        #     drive.publish(vel_msg)
        #     rospy.sleep(0.5)
            # vel_msg.linear.x = 0.1
            # vel_msg.angular.z = 1.17
            # drive.publish(vel_msg)
            rospy.sleep(1.6)
            #
            # vel_msg.linear.x = 0.1
            # vel_msg.angular.z = 0
            # drive.publish(vel_msg)
            # rospy.sleep(10.75)
            #
            # vel_msg.linear.x = 0.3
            # drive.publish(vel_msg)
            # rospy.sleep(4)




    plates.publish(plate_message(-1, "AA00"))

