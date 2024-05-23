# -*- coding: utf-8 -*-

#from IPython.display import display, Image
#import ipywidgets as widgets
import threading
import os
import pygame
from jetcam.utils import bgr8_to_jpeg
from jetcam.csi_camera import CSICamera

import cv2

os.environ["SDL_VIDEODRIVER"] = "dummy"

pygame.init()
pygame.joystick.init()

joystick = pygame.joystick.Joystick(0)
joystick.init()

camera = CSICamera(capture_width=1280, capture_height=720, downsample=2, capture_fps=30)

# 녹화할 영상의 코덱을 설정하기 위해 fourcc 변수를 초기화합니다. 여기서는 XVID 코덱을 사용합니다.
fourcc = cv2.VideoWriter_fourcc(*'XVID')

frame = camera.read()
video = cv2.VideoWriter("test_Traffic_straight.avi", fourcc, 20.0, (frame.shape[1], frame.shape[0]))

running = True
vid_capture = False
try:
    while running:
        pygame.event.pump()
        print(joystick.get_axis(1))
        if joystick.get_button(4): # start button
            print("Button pressed")
            vid_capture = True
        while vid_capture:
            frame = camera.read()
            video.write(frame)
            print('New frame written')
            
            if joystick.get_button(3): # start button
                vid_capture = False 
                print('finish')
            #pygame.event.pump()
            #if joystick.get_button(3):
finally:
    print("Capture done")
    video.release()
    running = False
