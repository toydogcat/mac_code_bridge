import cv2
import json
import time
import math
import numpy as np


class Manager:
    def __init__(self, video_path, state):
        # height, width, _ = img.shape
        cap              = cv2.VideoCapture(video_path)
        self.height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_fps   = cap.get(cv2.CAP_PROP_FPS)  # the video average fps
        self.frame_count = 0
        
        # warning
        self.warning_flag = False
        self.warning_num  = 1
        self.developer    = False
        
        self.state = state
        self.state_rate = 0.7 if state == "run" else 1.2
        
        self.normal_sleep_time = 1 / self.video_fps

        # Human info
        self.energy = 500.0
        self.heart_rate = 60
        self.resistance = 1
        self.power_frame_output = 0
        self.energy_recover = 0
        self.energy_cost = 0
        self.slope_ground = 0
        self.theta = 0
        self.slope_modify = 0 # machine can modify

        self.speed = 0
        self.acceleration = 0
        self.speed_rate = 1
        self.sleep_time = self.normal_sleep_time / self.speed_rate
        
        self.whiteboard = Whiteboard(100, self.width)
        
    
    def modify_power(self, x, y):
        self.power_frame_output = ( x + 100 + self.height - y ) / ( self.width + self.height )

    def update_parameter(self):
        energy_left = 500 - self.energy
        if self.energy >= 100:
            self.heart_rate = energy_left / 16 + 60
        else:
            self.heart_rate = (energy_left - 400) * (3 / 20) + 85

        self.energy_recover = math.log( self.heart_rate - 50 ) * 2 - 4
        self.theta = math.atan(self.slope_ground + self.slope_modify)
        self.acceleration = self.power_frame_output * self.state_rate * ( 1 - math.sin(self.theta) ) / ( self.resistance / 2 ) - 1

        # slope ground need to get from json file
        # self.theta 
        self.speed += self.acceleration
        self.speed  = np.clip(self.speed, -30, 30)
        if self.speed >= 0:
            self.speed_rate = self.speed * 0.1 + 1
        else:
            self.speed_rate = (30 + self.speed) * (1 / 40) + (1 / 4) 
        self.sleep_time = self.normal_sleep_time / self.speed_rate
        self.energy_cost = (self.speed + 30) / 10

        self.energy += (self.energy_recover - self.energy_cost)
        self.energy = np.clip(self.energy, 0, 500)
        self.warning_flag = True if self.energy < 100 else False

        # speed_rate



    def update(self, img):
        self.frame = img
        
        self.update_parameter()
        
        # write info to whiteboard
        self.whiteboard.new()
        
        text = f"speed : {round(self.speed, 2)}"
        cv2.putText(self.whiteboard.img, 
                    text, (30, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
        text = f"angle : {int(self.theta * 180 / math.pi)}"
        cv2.putText(self.whiteboard.img, 
                    text, (180, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
        text = f"resist : {self.resistance}"
        cv2.putText(self.whiteboard.img, 
                    text, (330, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)

        text = f"state : {self.state}"
        cv2.putText(self.whiteboard.img, 
                    text, (30, 70), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
        text = f"heart : {int(self.heart_rate)}"
        cv2.putText(self.whiteboard.img, 
                    text, (180, 70), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)

        if self.developer:
            text = f"normal : {int(self.normal_sleep_time*1000)}"
            cv2.putText(self.whiteboard.img, 
                        text, (900, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (130, 130,0), 1, cv2.LINE_AA)
            text = f"sleep : {int(self.sleep_time*1000)}"
            cv2.putText(self.whiteboard.img, 
                        text, (900, 35), cv2.FONT_HERSHEY_COMPLEX, 0.5, (130, 130,0), 1, cv2.LINE_AA)

            text = f"power : {round(self.power_frame_output, 4)}"
            cv2.putText(self.whiteboard.img, 
                        text, (1050, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (130, 130,0), 1, cv2.LINE_AA)
            text = f"energy : {round(self.energy, 4)}"
            cv2.putText(self.whiteboard.img, 
                        text, (1050, 35), cv2.FONT_HERSHEY_COMPLEX, 0.5, (130, 130,0), 1, cv2.LINE_AA)
            text = f"recover : {round(self.energy_recover, 4)}"
            cv2.putText(self.whiteboard.img, 
                        text, (1050, 55), cv2.FONT_HERSHEY_COMPLEX, 0.5, (130, 130,0), 1, cv2.LINE_AA)
            text = f"cost : {round(self.energy_cost, 4)}"
            cv2.putText(self.whiteboard.img, 
                        text, (1050, 75), cv2.FONT_HERSHEY_COMPLEX, 0.5, (130, 130,0), 1, cv2.LINE_AA)
            text = f"acc : {round(self.acceleration, 4)}"
            cv2.putText(self.whiteboard.img, 
                        text, (1050, 95), cv2.FONT_HERSHEY_COMPLEX, 0.5, (130, 130,0), 1, cv2.LINE_AA)
            
        
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
        

