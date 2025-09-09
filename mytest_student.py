# -*- coding: utf-8 -*-
import os
import cv2
import socketio
import base64
import shutil
import eventlet.wsgi
import sys
import numpy as np
import torch
from torch import nn
from flask import Flask
from PIL import Image
from io import BytesIO
from datetime import datetime

# socketio
sio = socketio.Server()

from mytrainer_student import PowerModeAutopilot


def preprocess(image):
    """
    preprocess the image
    :param image:
    :return:
    """
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image


def crop(image):
    return image[60:-25, :, :]


def resize(image):
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # current steering angle
        # steering_angle = float(data["steering_angle"])
        # current throttle
        # throttle = float(data["throttle"])
        # current speed
        speed = float(data["speed"])
        # center camera
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        try:
            image = np.asarray(image)  # from PIL image to numpy array
            image = preprocess(image)  # apply the preprocessing
            image = image.transpose((2, 0, 1))  # reshape to (channels, height, width)
            image = torch.from_numpy(image).float().unsqueeze(0)  # convert to PyTorch tensor

            # predict the steering angle
            steering_angle = float(model(image).item())

            # Adjust the throttle according to the speed,
            # if greater than the maximum speed to slow down,
            # if less than the minimum speed to increase acceleration
            if speed > MAX_SPEED:
                speed_limit = MIN_SPEED  # slow down
            else:
                speed_limit = MAX_SPEED
            throttle = 1.0 - steering_angle ** 2 - (speed / speed_limit) ** 2

            print('{} {} {}'.format(steering_angle, throttle, speed))
            send_control(steering_angle, throttle)
        except Exception as e:
            print(e)

        # save frame
        if image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(image_folder, timestamp)
            Image.fromarray(image.numpy().transpose((1, 2, 0)).astype(np.uint8)).save('{}.jpg'.format(image_filename))
    else:
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    model_path = sys.argv[1] # pass the model file path in your python arguments
    IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
    MAX_SPEED, MIN_SPEED = 25, 10
    # load the model trained with mytrainer.py
    model = PowerModeAutopilot()
    model.load_state_dict(torch.load(model_path))
    print('loaded ', model_path)
    model.eval()
    image_folder = ''
    # Set image cache directory
    if image_folder != '':
        if os.path.exists(image_folder):
            shutil.rmtree(image_folder)
        os.makedirs(image_folder)

    app = Flask(__name__)
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)