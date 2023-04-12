import torch
import torchvision
import utils
from models.experimental import attempt_load

from typing import Generator

import matplotlib.pyplot as plt
import numpy as np

import cv2

import matplotlib
matplotlib.use('TkAgg')

#%matplotlib inline 

# define the path to yolov7 model weight
weights_path = 'best_trained.pt'

def generate_frames(video_file: str) -> Generator[np.ndarray, None, None]:
    video = cv2.VideoCapture(video_file)

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        yield frame

    video.release()

def plot_img(image: np.ndarray, size: int = 12) -> None:
    plt.figure(figsize=(size, size))
    plt.imshow(image[...,::-1])
    plt.show()

video_path = 'street.mp4'
frame_iterator = iter(generate_frames(video_file = video_path))
frame = next(frame_iterator)

# plot_img(frame, 16) 

# convert the image(resized to input requirement) to tensor
frame = torchvision.transforms.functional.to_tensor(cv2.resize(frame, (640, 640)))

# Addd a batch dimension to the tensor
frame = frame.unsqueeze(0)

import torch
model = attempt_load(weights_path, map_location=torch.device('cpu'))

with torch.no_grad():
    output = model(frame)

