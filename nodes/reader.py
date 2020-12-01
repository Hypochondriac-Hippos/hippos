#!/usr/bin/env python2

"""
Subscribe to license plate image topics and itentify them with a neural network.
"""

import datetime
import time

import rospy
import sensor_msgs
import std_msgs
import cv_bridge

import read
import util

bridge = cv_bridge.CvBridge()


def process_frame(ros_image, score):
    frame = bridge.imgmsg_to_cv2(ros_image, "bgr8")
    prediction = read.predict(frame, debug=True)
    print("{} {}".format(datetime.datetime.now().isoformat(), prediction))
    if prediction is not None:
        score.publish(util.plate_message(prediction[0], prediction[1]))


if __name__ == "__main__":
    rospy.init_node("reader", anonymous=True)
    score = rospy.Publisher(util.topics.plates, std_msgs.msg.String, queue_size=1)
    rospy.Subscriber(
        util.topics.camera, sensor_msgs.msg.Image, process_frame, callback_args=score
    )

    time.sleep(1)

    rospy.spin()
