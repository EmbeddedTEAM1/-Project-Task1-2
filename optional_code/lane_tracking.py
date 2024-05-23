import cv2
import numpy as np
import math

from jetcam.utils import bgr8_to_jpeg
from jetcam.csi_camera import CSICamera

ROI_START_ROW = 140
ROI_END_ROW = 360
WIDTH = 640
ROI_HEIGHT = ROI_END_ROW - ROI_START_ROW
L_ROW = 100 # 차선 인식 기준
img_ready = False

def lane_drive(image):
    global img_ready
    prev_x_left = 0
    prev_x_right = WIDTH
    '''
    while img_ready == False:
        continue
    '''
    img = image.copy() 
    display_img = img  
    img_ready = False  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray,(5, 5), 0)
    edge_img = cv2.Canny(np.uint8(blur_gray), 60, 75)
    angle = 0
    roi_img = img[ROI_START_ROW:ROI_END_ROW, 0:WIDTH]
    
    roi_edge_img = edge_img[ROI_START_ROW:ROI_END_ROW, 0:WIDTH]
    # cv2.imshow("roi edge img", roi_edge_img)

    all_lines = cv2.HoughLinesP(roi_edge_img, 1, math.pi/180,150,50,20)
    
    if all_lines is None:
        return display_img, angle

    line_draw_img = roi_img.copy()
    
    slopes = []
    filtered_lines = []

    for line in all_lines:
        x1, y1, x2, y2 = line[0]

        if (x2 == x1):
            slope = 1000.0
        else:
            slope = float(y2-y1) / float(x2-x1)
    
        if 0.2 < abs(slope):
            slopes.append(slope)
            filtered_lines.append(line[0])

    if len(filtered_lines) == 0:
        return display_img, angle

    left_lines = []
    right_lines = []

    for j in range(len(slopes)):
        Line = filtered_lines[j]
        slope = slopes[j]

        x1,y1, x2,y2 = Line

        Margin = 0
        
        if (slope < 0) and (x2 < WIDTH/2-Margin):
            left_lines.append(Line.tolist())

        elif (slope > 0) and (x1 > WIDTH/2+Margin):
            right_lines.append(Line.tolist())

    # print("Number of left lines : %d" % len(left_lines))
    # print("Number of right lines : %d" % len(right_lines))

    line_draw_img = roi_img.copy()
    
    for line in left_lines:
        x1,y1, x2,y2 = line
        cv2.line(line_draw_img, (x1,y1), (x2,y2), (0,0,255), 2)

    for line in right_lines:
        x1,y1, x2,y2 = line
        cv2.line(line_draw_img, (x1,y1), (x2,y2), (0,255,255), 2)

    m_left, b_left = 0.0, 0.0
    x_sum, y_sum, m_sum = 0.0, 0.0, 0.0

    size = len(left_lines)
    if size != 0:
        for line in left_lines:
            x1, y1, x2, y2 = line
            x_sum += x1 + x2
            y_sum += y1 + y2
            if(x2 != x1):
                m_sum += float(y2-y1)/float(x2-x1)
            else:
                m_sum += 0                
            
        x_avg = x_sum / (size*2)
        y_avg = y_sum / (size*2)
        m_left = m_sum / size
        b_left = y_avg - m_left * x_avg

        if m_left != 0.0:
            x1 = int((0.0 - b_left) / m_left)        
            x2 = int((ROI_HEIGHT - b_left) / m_left)

            cv2.line(line_draw_img, (x1,0), (x2,ROI_HEIGHT), (255,0,0), 2)

    m_right, b_right = 0.0, 0.0
    x_sum, y_sum, m_sum = 0.0, 0.0, 0.0

    size = len(right_lines)
    if size != 0:
        for line in right_lines:
            x1, y1, x2, y2 = line
            x_sum += x1 + x2
            y_sum += y1 + y2
            if(x2 != x1):
                m_sum += float(y2-y1)/float(x2-x1)
            else:
                m_sum += 0     

        x_avg = x_sum / (size*2)
        y_avg = y_sum / (size*2)
        m_right = m_sum / size
        b_right = y_avg - m_right * x_avg

        if m_right != 0.0:
            x1 = int((0.0 - b_right) / m_right)  
            x2 = int((ROI_HEIGHT - b_right) / m_right)

            cv2.line(line_draw_img, (x1,0), (x2,ROI_HEIGHT), (255,0,0), 2)

    if m_left == 0.0:
        x_left = prev_x_left	
    else:
        x_left = int((L_ROW - b_left) / m_left)
                        
    if m_right == 0.0:
        x_right = prev_x_right
    else:
        x_right = int((L_ROW - b_right) / m_right)

    if m_left == 0.0 and m_right != 0.0:
        x_left = x_right - 380

    if m_left != 0.0 and m_right == 0.0:
        x_right = x_left + 380
			
    prev_x_left = x_left
    prev_x_right = x_right
	
    x_midpoint = (x_left + x_right) // 2 
    view_center = WIDTH//2

    cv2.line(line_draw_img, (0,L_ROW), (WIDTH,L_ROW), (0,255,255), 2)
    cv2.rectangle(line_draw_img, (x_left-5,L_ROW-5), (x_left+5,L_ROW+5), (0,255,0), 4)
    cv2.rectangle(line_draw_img, (x_right-5,L_ROW-5), (x_right+5,L_ROW+5), (0,255,0), 4)
    cv2.rectangle(line_draw_img, (x_midpoint-5,L_ROW-5), (x_midpoint+5,L_ROW+5), (255,0,0), 4)
    cv2.rectangle(line_draw_img, (view_center-5,L_ROW-5), (view_center+5,L_ROW+5), (0,0,255), 4)

    display_img[ROI_START_ROW:ROI_END_ROW, 0:WIDTH] = line_draw_img

    angle = (x_midpoint-view_center) // 2
    
    return display_img, angle

if __name__ == '__main__':

    camera = CSICamera(capture_device=0, capture_width=1280, capture_height=720, downsample=2, capture_fps=30)
    print('Camera mounted')

    try:
        while True:
            image = camera.read()
            display_img, angle = lane_drive(image)
            
            cv2.imshow("Lanes positions", display_img)
            cv2.waitKey(1)
            
            
            print(angle)

    finally:
        camera.release()
