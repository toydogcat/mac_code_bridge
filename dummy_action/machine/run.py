import cv2
import time
import numpy as np
import argparse
import threading
from utils import tools



# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--state", type=str, default="run", help="run or ride.")
parser.add_argument("-vp", "--video_path", type=str, default=None, help="the video path.")
parser.add_argument("-vi", "--video_list_index", type=int, default=0, help="the video index in the video list.")
args = parser.parse_args()

video_path_list = [
    "datas/videos/MXBK1701_mxi01_30s.mp4",
    "/Users/owenchen/Movies/MXBK1701_mxi01.mp4"
]

video_path = args.video_path if args.video_path else video_path_list[args.video_list_index] 
    
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
        manager.modify_energy()
    if event == 2 and flags == 2: # 右健
        manager.modify_state("ride") if manager.state == "run" else manager.modify_state("run")
        if manager.developer:
            manager.energy_water = 500
            
    
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
    if keycode == ord('z'):
        manager.developer = False if manager.developer == True else True
    if keycode == 0:  # 按下鍵盤的「上」
        manager.slope_modify_level = np.clip(manager.slope_modify_level + 1, 0, 10) 
    if keycode == 1:  # 按下鍵盤的「下」
        manager.slope_modify_level = np.clip(manager.slope_modify_level - 1, 0, 10) 
    if keycode == 2:  # 按下鍵盤的「左」
        manager.resistance = np.clip(manager.resistance - 1, 1, 10) 
    if keycode == 3:  # 按下鍵盤的「右」
        manager.resistance = np.clip(manager.resistance + 1, 1, 10) 
    
    
cap.release()
cv2.destroyAllWindows()



