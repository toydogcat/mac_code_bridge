import cv2
import threading
from utils import tools

video_path = "datas/videos/MXBK1701_mxi01_30s.mp4"
image_path = "output.jpg"
keyboard_input = ""
mouse_input = ""

image_bgr = cv2.imread(image_path)
manager = tools.Manager(video_path)
cap = cv2.VideoCapture(video_path)

# setting mouse event
def show_xy(event,x,y,flags, userdata):
    print(event, x, y, flags, userdata)

cv2.imshow('live', image_bgr)
cv2.setMouseCallback('live', show_xy, 'toby')  


while cap.isOpened():
    # ret, frame = cap.read()
    # if frame is read correctly ret is True
    # if not ret:
    #     print("Can't receive frame (stream end?). Exiting ...")
    #     break
    
    # setting keyboard event
    keycode = cv2.waitKey(0)
    if keycode == ord('q') or keycode == 27:
        break
    if keycode == 0:  # 按下鍵盤的「上」
        keyboard_input = "up"
        print(f"keyboard : {keyboard_input}")
    if keycode == 1:  # 按下鍵盤的「下」
        keyboard_input = "down"
        print(f"keyboard : {keyboard_input}")
    if keycode == 2:  # 按下鍵盤的「右」
        keyboard_input = "right"
        print(f"keyboard : {keyboard_input}")
    if keycode == 3:  # 按下鍵盤的「左」
        keyboard_input = "left"
        print(f"keyboard : {keyboard_input}")
    
    
cap.release()
cv2.destroyAllWindows()



