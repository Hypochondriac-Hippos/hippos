#!/usr/bin/env python

from collections import namedtuple
import cPickle
import os

import cv2

topics = namedtuple("Topics", ("camera", "clock", "drive", "plates", "warning"))(
    "/R1/pi_camera/image_raw",
    "/clock",
    "/R1/cmd_vel",
    "/license_plate",
    "/hippos/pedestrian",
)

plate_topics = namedtuple("PlateTopics", ("edges", "lines", "rects"))(
    "/hippos/plate/edges", "/hippos/plate/lines", "/hippos/plate/rects"
)


def plate_message(location, plate_number):
    """Format a message for the license plate scorer"""
    return "Hippos,MitiFizz,{},{}".format(location, plate_number)


def imread(file, *args, **kwargs):
    image = cv2.imread(file, *args, **kwargs)
    if image is None:
        raise IOError("Couldn't read file {}. CWD is {}".format(file, os.getcwd()))
    return image


def stringify(obj):
    """Return a serialization of the object that's safe for a ROS string message"""
    return cPickle.dumps(obj, 2)


def destringify(obj):
    """Deserialize an object serialized with stringify."""
    return cPickle.loads(obj.data)
