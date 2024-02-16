import cv2
import time
import numpy as np
import argparse
import threading
from utils import tools



# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--state", type=str, default="run", help="run or ride.")
parser.add_argument("-vp", "--video_path", type=str, default="datas/videos/MXBK1701_mxi01_30s.mp4", help="the video path.")
args = parser.parse_args()



video_path = args.video_path 
image_path = "output.jpg"
# keyboard_input = ""
# mouse_input = ""
start_time = time.time()

manager = tools.Manager(video_path, args.state)

image_bgr = cv2.imread(image_path)
cap = cv2.VideoCapture(video_path)

cv2.imshow('live', image_bgr)

# setting mouse event
# def show_xy(event,x,y,flags, userdata):
#     print(event, x, y, flags, userdata)
# cv2.setMouseCallback('live', show_xy, 'toby')  

def mouse_event(event, x, y, flags, userdata):
    global manager
    manager.modify_power(x, y)
    
    if event == 1 and flags == 1: # 左健
        manager.developer = False
    if event == 2 and flags == 2: # 右健
        manager.developer = True
cv2.setMouseCallback('live', mouse_event) 


while cap.isOpened():
    end_time = time.time()

    if (end_time - start_time) > manager.sleep_time:
        ret, frame = cap.read()
        if not ret:
            print("Video End. Exiting ...")
            break
        manager.update(frame)
        cv2.imshow('live', manager.image_out)
        start_time = time.time()
    
    # setting keyboard event
    keycode = cv2.waitKey(1)
    if keycode == ord('q') or keycode == 27:
        break
    if keycode == 0:  # 按下鍵盤的「上」
        manager.slope_modify = np.clip(manager.slope_modify + 0.02, 0, 0.2) 
    if keycode == 1:  # 按下鍵盤的「下」
        manager.slope_modify = np.clip(manager.slope_modify - 0.02, 0, 0.2) 
    if keycode == 2:  # 按下鍵盤的「左」
        manager.resistance = np.clip(manager.resistance - 1, 1, 10) 
    if keycode == 3:  # 按下鍵盤的「右」
        manager.resistance = np.clip(manager.resistance + 1, 1, 10) 
    
    
cap.release()
cv2.destroyAllWindows()



