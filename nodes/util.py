#!/usr/bin/env python

from collections import namedtuple
import os

import cv2

topics = namedtuple("Topics", ("camera", "clock", "drive", "plates"))(
    "/R1/pi_camera/image_raw", "/clock", "/R1/cmd_vel", "/license_plate"
)


def plate_message(location, plate_number):
    """Format a message for the license plate scorer"""
    return "Hippos,MitiFizz,{},{}".format(location, plate_number)


def imread(file, *args, **kwargs):
    image = cv2.imread(file, *args, **kwargs)
    if image is None:
        raise IOError("Couldn't read file {}. CWD is {}".format(file, os.getcwd()))
    return image
