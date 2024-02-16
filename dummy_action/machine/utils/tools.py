import cv2
import json
import time
import numpy as np


class Manager:
    def __init__(self, video_path, state):
        # height, width, _ = img.shape
        cap            = cv2.VideoCapture(video_path)
        self.height    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_fps = cap.get(cv2.CAP_PROP_FPS)  # the video average fps
        
        # warning
        self.warning_flag = False
        self.warning_num = 1
        self.developer = False
        
        self.state = state
        self.state_rate = 1 if state == "run" else 1.5
        
        self.normal_sleep_time = 1 / self.video_fps
        self.heart_rate = 60
        self.resistance = 1
        self.power_frame_output = 0
        
        
        self.whiteboard = Whiteboard(100, self.width)
        # Human info
        self.energy = 500.0
    
    def modify_power(self, x, y):
        self.power_frame_output = ( x + 100 + self.height - y ) / ( self.width + self.height )
    
    def update(self, img):
        self.frame = img
        
        # value check
        self.energy = np.clip(self.energy, 0, 500)
        
        # write info to whiteboard
        self.whiteboard.new()
        
        text = f"state : {self.state}"
        cv2.putText(self.whiteboard.img, text, (30, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
        
        if self.developer:
            text = f"power : {round(self.power_frame_output, 3)}"
            cv2.putText(self.whiteboard.img, text, (1000, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
        
        # concat images
        self.image_out = cv2.vconcat([self.whiteboard.img, self.frame])
        if self.warning_flag:
            h, w, _ = self.image_out.shape
            cv2.rectangle(self.image_out, (0,0), (w,h), (0,0,int(255/self.warning_num)), 10)
            self.warning_num = self.warning_num % 10 + 1  
        
        
        
        
        
class Whiteboard:
    def __init__(self, height, width):
        self.height = height
        self.width  = width
        self.img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
    def new(self):
        self.img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        

