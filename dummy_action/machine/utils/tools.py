import cv2
import json
import time
import math
import random
import numpy as np
from pathlib import Path


class Manager:
    def __init__(self, video_path, state, record_flag=True):
        # height, width, _ = img.shape
        cap                  = cv2.VideoCapture(video_path)
        self.height          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_fps       = cap.get(cv2.CAP_PROP_FPS)  # the video average fps
        self.record_flag     = record_flag
        self.frame_count_all = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count     = 0
        if self.record_flag:
            # 錄製參數設定
            fps = 15
            frame_size = (1280, 820)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter('record/output_demo.mp4', fourcc, fps, frame_size)
        
        # deal with npz or json
        # "datas/videos/MXBK1701_mxi01_30s.mp4"
        _path = Path(video_path)
        json_path = _path.parents[1] / 'json' / (_path.name.split('.')[0]+'.json')
        # npz_path  = _path.parents[1] / 'npz' / (_path.name.split('.')[0]+'.npz')
        
        if json_path.exists():
            with open(json_path, 'r') as jsonfile:
                self.json_data = json.load(jsonfile)
        else:
            self.json_data = None
            print("we can't find the json file for the video topography.")
        
        # warning
        self.warning_flag = False
        self.warning_num  = 1
        self.developer    = False
        
        self.modify_state(state)
        # self.state = state
        # self.state_rate = 0.7 if state == "run" else 1.5
        
        self.normal_sleep_time = 1 / self.video_fps
        self.spend_time = 0

        # Human info
        self.energy = 500.0
        self.energy_water = 5
        self.drink_water = 0
        
        self.heart_rate = 60
        self.resistance = 5
        
        self.power_frame_output = 0  # 心願出力
        self.power = 0               # 真實出力
        
        self.energy_recover = 0
        self.energy_cost = 0
        self.slope_ground = 0        # load from json file
        self.theta = 0
        
        self.slope_modify_level = 0  # machine can modify
        self.slope_modify = self.slope_modify_level * 0.02 

        self.speed = -30
        self.acceleration = 0
        self.speed_rate = 1
        self.sleep_time = self.normal_sleep_time / self.speed_rate
        
        self.whiteboard = Whiteboard(100, self.width)
        
    def modify_state(self, state):
        self.state = state
        self.state_rate = 0.7 if state == "run" else 1.5
        
    def modify_energy(self, x, y):
        self.drink_water = 5  
        diff_x = abs(random.randint(0, 1280) - x)
        diff_y = abs(random.randint(0, 820) - y)
        random_water = 200 if (diff_x + diff_y) == 0 else 100 / (diff_x + diff_y)
        if self.developer:
            print(f"water : {self.energy_water + random_water}")
        self.energy = np.clip(self.energy + self.energy_water + random_water, 0, 500)
        
    def modify_power(self, x, y):
        self.power_frame_output = 5 * ( 1 - abs(x / self.width - 0.5) - abs(y / self.height - 0.5) )
        
    def modify_spend_time(self, spend_time):
        self.spend_time = spend_time

    def update_parameter(self):
        energy_left = 500 - self.energy
        if self.energy >= 100:
            self.heart_rate = energy_left / 16 + 60
            random_rate = 2
        else:
            self.heart_rate = (energy_left - 400) * (3 / 20) + 85
            random_rate = 10

        self.energy_recover = math.log( self.heart_rate - 50 ) * 2 - self.power - random_rate * random.random()
        
        self.power = self.power_frame_output * math.sqrt(self.energy/100) if self.energy < 100 else self.power_frame_output
        self.energy_cost = self.power + (self.speed + 30) / 100
        
        self.energy += (self.energy_recover - self.energy_cost)
        self.energy = np.clip(self.energy, 0, 500)
        self.warning_flag = True if self.energy < 100 else False
        
        if self.json_data:
            json_mask_rate = self.json_data[self.frame_count]["rate"] if self.frame_count < self.frame_count_all else self.json_data[-1]["rate"]
            self.slope_ground = (json_mask_rate - 0.5) * 10 if abs(json_mask_rate - 0.5) < 0.2 else 0
            
        
        self.slope_modify = self.slope_modify_level * 0.02
        self.theta = math.atan(self.slope_ground + self.slope_modify)
        self.acceleration = (self.power * self.state_rate * ( 1 - math.sin(self.theta) ) / ( self.resistance / 2 ) - 1) / 5

        # slope ground need to get from json file
        # self.theta 
        self.speed += self.acceleration
        self.speed  = np.clip(self.speed, -30, 30)
        if self.speed >= 0:
            self.speed_rate = self.speed * 0.1 + 1
        else:
            self.speed_rate = (30 + self.speed) * (1 / 40) + (1 / 4) 
        self.sleep_time = self.normal_sleep_time / self.speed_rate
        
        if self.drink_water > 0:
            self.drink_water -= 1
        



    def update(self, img):
        self.frame_count += 1
        self.frame = img
        self.update_parameter()
        
        # write info to whiteboard
        self.whiteboard.new()
        
        text = f"speed : {round(self.speed, 2)}"
        cv2.putText(self.whiteboard.img, 
                    text, (30, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
        text = f"power : {int(self.power * 20)}"
        cv2.putText(self.whiteboard.img, 
                    text, (180, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
        text = f"angle : {int(self.theta * 180 / math.pi)}"
        cv2.putText(self.whiteboard.img, 
                    text, (330, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
        text = f"resist : {self.resistance}"
        cv2.putText(self.whiteboard.img, 
                    text, (480, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
        text = f"process : {int(100*self.frame_count/self.frame_count_all)} %"
        cv2.putText(self.whiteboard.img, 
                    text, (630, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)

        text = f"state : {self.state}"
        cv2.putText(self.whiteboard.img, 
                    text, (30, 70), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
        text = f"heart : {int(self.heart_rate)}"
        cv2.putText(self.whiteboard.img, 
                    text, (180, 70), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
        
        if self.drink_water > 0:
            cv2.putText(self.whiteboard.img, 
                    "Drinking Water", (330, 70), cv2.FONT_HERSHEY_COMPLEX, 0.7, (77,230,153), 1, cv2.LINE_AA)
            
        if self.developer:
            text = f"normal : {int(self.normal_sleep_time*1000)}"
            cv2.putText(self.whiteboard.img, 
                        text, (900, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (130, 130,0), 1, cv2.LINE_AA)
            text = f"sleep : {int(self.sleep_time*1000)}"
            cv2.putText(self.whiteboard.img, 
                        text, (900, 35), cv2.FONT_HERSHEY_COMPLEX, 0.5, (130, 130,0), 1, cv2.LINE_AA)
            text = f"spend : {int(self.spend_time*1000)}"
            cv2.putText(self.whiteboard.img, 
                        text, (900, 55), cv2.FONT_HERSHEY_COMPLEX, 0.5, (130, 130,0), 1, cv2.LINE_AA)

            text = f"ipower : {round(self.power_frame_output, 4)}"
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
        
        # record
        if self.record_flag:
            self.out.write(self.image_out)
            
        
        
        
        
        
class Whiteboard:
    def __init__(self, height, width):
        self.height = height
        self.width  = width
        self.img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
    def new(self):
        self.img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        

