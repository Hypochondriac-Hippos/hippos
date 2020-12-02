#!/usr/bin/env python2

"""
Subscribe to license plate image topics and itentify them with a neural network.
"""

import datetime
import time

import rospy
import std_msgs
import cv_bridge

import read
import util

bridge = cv_bridge.CvBridge()


def process_frame(in_msg, score):
    lines, edges, bottom = util.destringify(in_msg)
    prediction = read.predict(lines, edges, bottom, debug=True)

    print("{} {}".format(datetime.datetime.now().isoformat(), prediction))
    if prediction is not None:
        score.publish(util.plate_message(prediction[0], prediction[1]))


if __name__ == "__main__":
    rospy.init_node("reader", anonymous=True)
    score = rospy.Publisher(util.topics.plates, std_msgs.msg.String, queue_size=1)
    rospy.Subscriber(
        util.plate_topics.lines, std_msgs.msg.String, process_frame, callback_args=score
    )

    time.sleep(1)

    rospy.spin()
