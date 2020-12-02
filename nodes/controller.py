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


def move(vel_msg, x=0.0,y=0.0,z=0.0,xR=0.0,yR=0.0,zR=0.0):

    vel_msg.linear.x = x
    vel_msg.linear.y = y
    vel_msg.linear.z = z
    vel_msg.angular.x = xR
    vel_msg.angular.y = yR
    vel_msg.angular.z = zR


if __name__ == "__main__":
    rospy.init_node("controller", anonymous=True)
    plates = rospy.Publisher(topics.plates, std_msgs.msg.String, queue_size=1)
    drive = rospy.Publisher(topics.drive, Twist, queue_size=1)

    rate = rospy.Rate(0.5)

    time.sleep(1)
    vel_msg = Twist()
    plates.publish(plate_message(0, "AA00"))
    while not rospy.is_shutdown():
        vel_msg.linear.x = 0.25
        vel_msg.angular.z = -0.05
        drive.publish(vel_msg)
        rate.sleep()



    plates.publish(plate_message(-1, "AA00"))

