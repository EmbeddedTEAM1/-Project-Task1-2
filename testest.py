import cv2
import numpy as np
import PIL.Image
import torch
import torch
import torchvision
from cnn.center_dataset import CenterDataset
import torch.nn.functional as f
from cnn.center_dataset import TEST_TRANSFORMS
import copy
from glob import glob
import os

# 이미지 파일 포맷과 주석 파일 경로 설정
img_filename_fmt = './dataset/images_edgeedge/'
ann_filename = 'dataset/annotation_edge_left.txt'

def get_model():
    model = torchvision.models.alexnet(num_classes=2, dropout=0.0)
    return model
def preprocess(image):
    device = torch.device('cuda')    
    image = TEST_TRANSFORMS(image).to(device)
    return image[None, ...]

device = torch.device('cuda')
model = get_model()
model.load_state_dict(torch.load('road_model_crop_edge_left.pth'))
model = model.to(device)

'''
# 주석 데이터 로드
with open(ann_filename, 'r') as f:
    data = [line.split() for line in f.readlines()]

# 모든 이미지에 대해 처리 반복
for index, (filename, xpos, ypos) in enumerate(data):
    xpos = int(xpos)
    ypos = int(ypos)

    # 이미지 로드
    image_ori = cv2.imread(filename)
    image_crop = image_ori[140:,:]
    height, width = image_ori.shape[:2]

    # 모델 예측 수행 (torch 필요)
    with torch.no_grad():
        image = preprocess(PIL.Image.fromarray(image_crop))  # 이미지 전처리
        output = model(image).detach().cpu().numpy()  # 모델 추론
    x, y = output[0]
    print(output)
    print(x)
    print(y)
    # 예측 좌표 조정
    x = (x / 2 + 0.5) * width
    y = (y / 2 + 0.5) * (height+140)
    
    
'''

input_images = glob(os.path.join(img_filename_fmt,"*.jpg"))
input_images.sort()
#print(input_images)

for filename in input_images:
    
    image_ori = cv2.imread(os.path.join(filename))
    
    image_crop = image_ori[140:,:]
    height, width = image_ori.shape[:2]

    # 모델 예측 수행 (torch 필요)
    with torch.no_grad():
        image = preprocess(PIL.Image.fromarray(image_crop))  # 이미지 전처리
        output = model(image).detach().cpu().numpy()  # 모델 추론
    x, y = output[0]
    print(output)
    print(x)
    print(y)
    # 예측 좌표 조정
    x = (x / 2 + 0.5) * width
    y = (y / 2 + 0.5) * (height-140)
    
    # 이미지 표시
    image_np = copy.deepcopy(np.asarray(image_crop))
    cv2.circle(image_np, (int(x), int(y)), radius=5, color=(255, 0, 0))  # 모델 예측 위치
    #cv2.circle(image_np, (xpos, ypos), radius=5, color=(0, 255, 0))     # 실제 GT 위치
    cv2.imshow('image',image_np)
    cv2.waitKey(1)

    # 필요하다면 중간에 일시정지
    input("Press Enter to continue to the next image...")
