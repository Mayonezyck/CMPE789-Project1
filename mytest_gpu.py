# -*- coding: utf-8 -*-
import os
import sys
import cv2
import base64
import shutil
import numpy as np
import torch
from torch import nn
from flask import Flask
from PIL import Image
from io import BytesIO
from datetime import datetime

import socketio
import eventlet.wsgi

# -----------------------------------------------------------------------------
# Device / perf
# -----------------------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True  # good for fixed-size inputs

# -----------------------------------------------------------------------------
# Simulator + model imports
# -----------------------------------------------------------------------------
from mytrainer_student import PowerModeAutopilot

# Socket.IO server (eventlet backend)
sio = socketio.Server(async_mode='eventlet', cors_allowed_origins='*')

# -----------------------------------------------------------------------------
# Constants (match training)
# -----------------------------------------------------------------------------
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
MAX_SPEED, MIN_SPEED = 25, 10    # simple speed policy caps

# Folder to optionally save processed frames ('' to disable)
image_folder = ''   # e.g., 'run_images'

# -----------------------------------------------------------------------------
# Preprocessing (match training pipeline)
# -----------------------------------------------------------------------------
def preprocess(image: np.ndarray) -> np.ndarray:
    """
    image: RGB uint8 or float array (H, W, 3)
    returns: float32 in [0,1], YUV colorspace, cropped & resized
    """
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)

    # enforce float32 [0,1] scaling
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    if image.max() > 1.5:  # likely uint8 range
        image /= 255.0

    return image

def crop(image: np.ndarray) -> np.ndarray:
    # keep rows [60 : -25], remove sky and hood (same as training)
    return image[60:-25, :, :]

def resize(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)

def rgb2yuv(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

# -----------------------------------------------------------------------------
# Telemetry handlers
# -----------------------------------------------------------------------------
@sio.on('telemetry')
def telemetry(sid, data):
    if not data:
        sio.emit('manual', data={}, skip_sid=True)
        return

    try:
        # current speed
        speed = float(data.get("speed", 0.0))

        # decode center camera image (RGB)
        pil_img = Image.open(BytesIO(base64.b64decode(data["image"])))
        img_np = np.asarray(pil_img)                 # H x W x 3 (RGB), uint8
        img_np = preprocess(img_np)                  # float32, [0,1], YUV
        chw = img_np.transpose(2, 0, 1)              # C x H x W

        # tensor on the proper device
        x = torch.from_numpy(chw).unsqueeze(0).to(DEVICE, dtype=torch.float32)

        # predict steering on GPU/CPU without grads
        with torch.no_grad():
            steering_angle = float(model(x).item())

        # sanitize and clamp steering
        if not np.isfinite(steering_angle):
            steering_angle = 0.0
        steering_angle = max(min(steering_angle, 1.0), -1.0)

        # throttle policy with startup push
        speed_limit = MIN_SPEED if speed > MAX_SPEED else MAX_SPEED
        if speed < 2.0:
            throttle = 0.35
        else:
            throttle = 1.0 - steering_angle**2 - (speed / speed_limit)**2
        throttle = max(min(throttle, 0.8), -0.2)

        print(f'{steering_angle:.4f} {throttle:.4f} {speed:.2f}')
        send_control(steering_angle, throttle)

        # optional: save preprocessed frames as jpgs
        if image_folder:
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            out_path = os.path.join(image_folder, f'{timestamp}.jpg')
            # convert back to uint8 RGB-like for inspection (we used YUV for the net,
            # but for quick logging it's fine to store the preprocessed YUV; if you prefer
            # RGB, save the original decoded image instead of img_np).
            save_img = (np.clip(img_np, 0.0, 1.0) * 255.0).astype(np.uint8)
            Image.fromarray(save_img).save(out_path)

    except Exception as e:
        print("telemetry error:", e)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={'steering_angle': str(steering_angle), 'throttle': str(throttle)},
        skip_sid=True
    )

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python drive_gpu.py <best_model.pth>")
        sys.exit(1)

    model_path = sys.argv[1]

    # Prepare optional image log dir
    if image_folder:
        if os.path.exists(image_folder):
            shutil.rmtree(image_folder)
        os.makedirs(image_folder, exist_ok=True)

    # Build the same architecture and load weights to DEVICE
    model = PowerModeAutopilot().to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    print('Loaded', model_path, 'on', DEVICE)
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))

    # Start server
    app = Flask(__name__)
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
