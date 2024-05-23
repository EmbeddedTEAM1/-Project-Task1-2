import cv2
import os
import numpy as np
import copy
import pygame
import PIL.Image
import torch
import torchvision
from torchvision.transforms import functional as F
from collections import deque
from jetcam.utils import bgr8_to_jpeg
from jetcam.csi_camera import CSICamera
from jetracer.nvidia_racecar import NvidiaRacecar
from cnn.center_dataset import TEST_TRANSFORMS
import argparse
from ultralytics import YOLO
import torch.nn as nn

stop_time = 0
frame_count = 0

class PIDController:
    """ PID 컨트롤러 구현 """
    def __init__(self, kp, kd):
        self.kp = kp  # 비례 게인
        #self.ki = ki  # 적분 게인
        self.kd = kd  # 미분 게인
        #self.integral = 0
        self.previous_error = 0
    
    def update(self, error, dt=1):
        """ PID 계산을 수행하고 스티어링 값을 반환 """
        #self.integral += error * dt
        #print(self.integral)
        #if abs(self.integral) >= 200:
        #    self.integral = 0
        derivative = (error - self.previous_error) / dt
        self.previous_error = error
        print("비례 :",self.kp * error ,"미분:", self.kd * derivative)
        return self.kp * error + self.kd * derivative

def preprocess(image):
    device = torch.device('cuda')    
    image = TEST_TRANSFORMS(image).to(device)
    return image[None, ...]

def get_model():
    model = torchvision.models.alexnet(num_classes=2, dropout=0.0)
    return model

class LaneFollower:
    def __init__(self):
        self.racecar = NvidiaRacecar()
        self.pid = PIDController(0.004, 0.0005)  # PID 컨트롤러 초기화
        self.steering_history = deque(maxlen=1)  # 스티어링 이력 저장
        self.direction_history = deque(maxlen=5)
        self.throttle_offset = 0
        self.racecar.throttle_gain = 0.43 #완충
        self.throttle_reduction = 0
        self.racecar.steering_offset = 0.05
        

    def run(self):
        global stop_time
        global frame_count
        plus_thr,minus_thr = False, False
        vid_capture = False
        frame = camera.read()
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter("test_edge_case1.avi", fourcc, 20.0, (frame.shape[1], frame.shape[0]))
        left_detect = False
        right_detect = False
        straight_detect = False
        buffer = False
        bus_detect = False
        cross_detect = False
        dir_traffic = False
        bus_traffic = False
        cross_traffic = False
        stop_time = 0
        buffer_time = 0
        
        throttle_std = 0.347# 스로틀링 기준값
        weight = 0.012
        
        try:
            while True:
                pygame.event.pump()
                
                frame = camera.read()
                
                frame_crop = frame[140:,:] # Crop image
                
                if vid_capture:
                    video.write(frame)
                    print('New frame written')
                frame_count += 1

                if frame_count%5 == 0:
                    bus_detect = False
                    cross_detect = False
                    dir_traffic = True
                    bus_traffic = False
                    cross_traffic = False
                    Object_classes = ['bus', 'cross', 'left', 'right', 'straight' ]
                    pred = yolo_model(frame,verbose = False)
                    for r in pred:
                        obj = r.boxes
                        for box in obj:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            area = abs(x1-x2)*abs(y1-y2)
                            #x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            score = round(float(box.conf[0]), 2)
                            label = int(box.cls[0])
                            cls_name = Object_classes[label]
                            # print("==============크기===========:",area , "==============확률===========:" ,score)
                            
                            if 10000> area > 3000 and score > 0.7:
                                cross_traffic = True
                            if area > 3000 and score > 0.5:
                                bus_traffic = True
                            if area > 1000 and score > 0.1:
                                dir_traffic = True
                                
                            if cls_name == 'bus' and bus_traffic:
                                bus_detect = True
                                #print("===========버스 감지 ================")
                            elif cls_name == 'cross' and cross_traffic and not buffer:
                                stop_time += 1
                                #print(stop_time)
                                #print("========= 횡단보도 감지 ================")
                                if stop_time > 10:
                                    cross_detect = False
                                    buffer = True
                                    stop_time = 0
                                    
                                    
                                else:
                                    cross_detect = True
                            
                            if buffer:
                                buffer_time += 1
                                print("<<<buffer>>>")
                                if buffer_time > 20:
                                    buffer = False
                                    buffer_time = 0
                                
                                    
                            elif cls_name == 'right' and dir_traffic:
                                # self.direction_history.append('right')
                                right_detect = True
                                left_detect = False
                                straight_detect = False
                            elif cls_name == 'left' and dir_traffic:
                                # self.direction_history.append('left')
                                left_detect = True
                                right_detect = False
                                straight_detect = False
                            elif cls_name == 'straight' and dir_traffic:
                                # self.direction_history.append('st')
                                straight_detect = True
                                left_detect = False
                                right_detect = False
                    
                with torch.no_grad():
                    frame_image = preprocess(PIL.Image.fromarray(frame_crop)) # crop frame
                    
                    if left_detect:
                        output = left_model(frame_image).detach().cpu().numpy()
                        print("=======LEFT=======")
                    elif straight_detect:
                        output = st_model(frame_image).detach().cpu().numpy()
                        print("======STRAIGHT======")
                    else:
                        output = right_model(frame_image).detach().cpu().numpy()
                        print("=======RiGFT=======")
                    
                x, y = output[0]
                width = frame.shape[1]
                height = frame.shape[0]

                x = (x / 2 + 0.5) * width

                y = (y / 2 + 0.5) * height
                center_x = width // 2
                error = center_x - x
                self.steering_history.append(error)
                avg_error = sum(self.steering_history) / len(self.steering_history)
                steering_angle = self.pid.update(avg_error)
                #print("앵글",steering_angle)
                self.throttle_reduction = abs(steering_angle*weight)
                
                if cross_detect:
                    self.throttle_offset = 0
                elif bus_detect:
                    self.throttle_offset = throttle_std - 0.025
                else:
                    self.throttle_offset = throttle_std
                
                if buffer and buffer_time < 1:
                    self.throttle_offset = throttle_std + 0.02
                
                print('Error:',avg_error)
                #print('steering angle:',steering_angle)
                #self.racecar.steering_gain = -0.65
                
                if avg_error > 85:
                    self.racecar.steering_gain = -0.99 + self.racecar.steering_offset
                    self.racecar.steering = -1
                    
                elif avg_error < -85:
                    self.racecar.steering_gain = -0.99 - self.racecar.steering_offset
                    self.racecar.steering = 1
                    
                else:
                    self.racecar.steering_gain = -1
                    self.racecar.steering = -steering_angle
                
                #print(self.racecar.steering_gain)
                
                if bus_detect:
                    print("[[[[[BUS DETECT]]]]]")
                    print(score,area)
                if cross_detect:
                    print("[[[[Cross DETECT]]]]")
                    print(score,area)
                    print(stop_time)
                if buffer:
                    print("[[[[BUFFER]]]]")
                
                #print(self.racecar.steering)
                self.racecar.throttle = self.throttle_offset - self.throttle_reduction


                #print('최종 조향각 :',self.racecar.steering*self.racecar.steering_gain + self.racecar.steering_offset)
                #print(self.throttle_reduction)
                #print(self.throttle_offset)
                #print(self.racecar.throttle)

                #print("스로틀게인",self.racecar.throttle_gain)
                #print("감속:", throttle_reduction)
                
                '''
                image_np = copy.deepcopy(np.asarray(frame_crop))
                cv2.circle(image_np, (int(x), int(y)), radius=5, color=(255, 0, 0))  # Pred
    
                cv2.imshow('camera',image_np)

                cv2.waitKey(1)
                '''
                
                
                if pygame.joystick.get_count() != 0:
                    if joystick.get_button(4): # start button
                        print("Button pressed")
                        vid_capture = True
                    if joystick.get_button(6):
                        plus_thr = True
                    if plus_thr == True and not joystick.get_button(6):
                        self.racecar.throttle_gain += 0.01
                        plus_thr = False
                        
                    if joystick.get_button(7):
                        minus_thr = True
                    if minus_thr == True and not joystick.get_button(7):
                        self.racecar.throttle_gain -= 0.01
                        minus_thr = False
                
                print("스로틀게인:",self.racecar.throttle_gain,plus_thr,minus_thr)
                os.system('clear')
                
        finally:
            print("The end")
            self.racecar.throttle = 0
            self.racecar.steering = 0
            video.release()
            camera.release()

if __name__ == '__main__':
    
    
    camera = CSICamera(capture_device=0, capture_width=1280, capture_height=720, downsample=2, capture_fps=30)
    print('Camera mounted')

    device = torch.device('cuda')
    st_model = get_model()
    st_model.load_state_dict(torch.load('road_model_109109_straight.pth'))
    st_model = st_model.to(device)
    print('Straight model mounted')
    
    left_model = get_model()
    left_model.load_state_dict(torch.load('road_model_109109109_left.pth'))
    left_model = left_model.to(device)
    print('Left model mounted')
    
    right_model = get_model()
    right_model.load_state_dict(torch.load('road_model_crop_edge_right.pth'))
    right_model = right_model.to(device)
    print('Right model mounted')
    
    yolo_model = YOLO("yolo_best.pt",  task='detect')
    print('YOLO model mounted')
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    pygame.joystick.init()
    
    if pygame.joystick.get_count() != 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print("Joystick connected")
    else:
        print("=== NO JOYSTICK ===")  
    follower = LaneFollower()
    follower.run()
