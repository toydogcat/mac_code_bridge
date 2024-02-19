import cv2
import time
import argparse
import numpy as np
from queue import Queue

# Arguments Bool Parameter
#   default : false 
#   add --state : true
# parser.add_argument("-s", "--state", action='store_true', help="run or ride.") 
#   default : none
#   add --state : true
# parser.add_argument("-s", "--state", action=argparse.BooleanOptionalAction, help="run or ride.")
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--flag", action='store_true', help="the flag for using display rate.") 
parser.add_argument("-ci", "--camera_index", type=int, default=0, help="The camera index.")
args = parser.parse_args()

# capture
cap = cv2.VideoCapture(args.camera_index)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

ret, frame = cap.read()
heigh, width, _ = frame.shape

# display_rate = 3/2:
# display_rate = 2/3:
if heigh == 1080 and width == 1920 and args.flag == False:
    display_rate = 2/3
else:
    display_rate = 1

queue_time = Queue(maxsize = 100)

# 錄製參數設定
fps = 25
frame_size = (int(width*display_rate), int(heigh*display_rate))
# 建立 VideoWriter 物件
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('../../record/camera_record.mp4', fourcc, fps, frame_size)


begin_time = time.time()
start_time = time.time()
while(True):
    # 擷取影像 To capture images
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    frame = cv2.flip(frame, 1)
    if display_rate != 1:
        frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)
    
    heigh, width, _ = frame.shape
    
    using_time = time.time() - begin_time
    if using_time < 20 :
        if queue_time.full():
            queue_time.get()
        queue_time.put(time.time() - start_time)
        text = f"resolution : {heigh, width}, fps : {int(1/np.array(queue_time.queue).mean())}"
        _color = (min(int(25.5 * (20 - using_time)), 255) ,0,0)
        
        cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, _color, 1, cv2.LINE_AA)
        start_time = time.time()
    
    
    # 顯示圖片 display images
    out.write(frame)
    cv2.imshow('live_show', frame)
    
    # 按下 q 鍵離開迴圈 Press the Q key to leave the loop
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()

