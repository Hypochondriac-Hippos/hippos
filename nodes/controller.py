#!/usr/bin/env python2

from collections import namedtuple
import time

import rospy
from geometry_msgs.msg import Twist
import std_msgs

topics = namedtuple("Topics", ("camera", "clock", "drive", "plates"))(
    "/RP1/pi_camera/image_raw", "/clock", "/R1/cmd_vel", "/license_plate"
)


def plate_message(location, plate_number):
    """Format a message for the license plate scorer"""
    return "Hippos,MitiFizz,{},{}".format(location, plate_number)


if __name__ == "__main__":
    rospy.init_node("time_trials", anonymous=True)
    plates = rospy.Publisher(topics.plates, std_msgs.msg.String, queue_size=1)
    drive = rospy.Publisher(topics.drive, Twist, queue_size=1)

    time.sleep(1)

    plates.publish(plate_message(0, "AA00"))

    move = Twist()
    move.linear.x = 0.75
    move.angular.z = 3
    drive.publish(move)

    rospy.sleep(0.5)
    move.angular.z = 0
    drive.publish(move)

    rospy.sleep(1.5)
    move.linear.x = 0
    move.angular.z = 0
    drive.publish(move)

    plates.publish(plate_message(-1, "AA00"))
