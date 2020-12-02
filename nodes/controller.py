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

topics = namedtuple("Topics", ("camera", "clock", "drive", "plates"))(
    "/R1/pi_camera/image_raw", "/clock", "/R1/cmd_vel", "/license_plate"
)


def plate_message(location, plate_number):
    """Format a message for the license plate scorer"""
    return "Hippos,MitiFizz,{},{}".format(location, plate_number)


if __name__ == "__main__":
    rospy.init_node("controller", anonymous=True)
    plates = rospy.Publisher(topics.plates, std_msgs.msg.String, queue_size=1)
    drive = rospy.Publisher(topics.drive, Twist, queue_size=1)

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

    vel_msg.linear.x = 0.1
    vel_msg.angular.z = 0
    drive.publish(vel_msg)
    rospy.sleep(11.1)

    while not rospy.is_shutdown():
        vel_msg.linear.x = 0.1
        vel_msg.angular.z = 1.17
        drive.publish(vel_msg)
        rospy.sleep(1.6)

        vel_msg.linear.x = 0.1
        vel_msg.angular.z = 0
        drive.publish(vel_msg)
        rospy.sleep(10.75)

        vel_msg.linear.x = 0.3
        drive.publish(vel_msg)
        rospy.sleep(4)




    plates.publish(plate_message(-1, "AA00"))

