#!/usr/bin/env python2

"""
Detect pedestrians
"""

import time

import cv2
import cv_bridge
import numpy as np
import rospy
import sensor_msgs
import std_msgs

import util

bridge = cv_bridge.CvBridge()
warning = rospy.Publisher("/hippos/pedestrian", std_msgs.msg.String, queue_size=1)
red_pub = rospy.Publisher("/hippos/debug/red", sensor_msgs.msg.Image, queue_size=1)
feature_pub = rospy.Publisher(
    "/hippos/debug/pedestrian", sensor_msgs.msg.Image, queue_size=1
)


def process_frame(ros_image, debug=True):
    frame = bridge.imgmsg_to_cv2(ros_image)
    l, a, b = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB))
    red = a > 200
    if debug:
        display = red.astype(np.uint8) * 255
        cv2.putText(
            display,
            str(np.count_nonzero(red)),
            (0, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            255,
            10,
        )
        red_pub.publish(bridge.cv2_to_imgmsg(display))

    if np.count_nonzero(red) < 10:
        warning.publish(util.stringify(False))
    if np.count_nonzero(red) > 10:
        top = min(np.nonzero(red)[0])
        bottom = max(np.nonzero(red)[0])
        top_left = min(np.nonzero(red[top : top + 10])[1])
        top_right = max(np.nonzero(red[top : top + 10])[1])
        bottom_left = min(np.nonzero(red[bottom - 50 : bottom])[1])
        bottom_right = max(np.nonzero(red[bottom - 50 : bottom])[1])
        left = min(top_left, bottom_left)
        right = max(top_right, bottom_right)
        crosswalk = slice(top, bottom), slice(left, right)

        walk_mask = np.zeros_like(a)
        walk_mask = cv2.fillConvexPoly(
            walk_mask,
            np.asarray(
                [
                    [top_left, top],
                    [top_right, top],
                    [bottom_right, bottom],
                    [bottom_left, bottom],
                ],
                dtype=np.int32,
            ),
            255,
        )

        person_in_walk = np.logical_and(
            np.logical_and(walk_mask[crosswalk], b[crosswalk] > 130), a[crosswalk] > 100
        )

        warning.publish(util.stringify(person_in_walk.any()))

        if debug:
            feature_pub.publish(
                bridge.cv2_to_imgmsg((person_in_walk).astype(np.uint8) * 255)
            )


if __name__ == "__main__":
    rospy.init_node("pedestrian", anonymous=True)
    rospy.Subscriber(util.topics.camera, sensor_msgs.msg.Image, process_frame)
    time.sleep(1)
    rospy.spin()
