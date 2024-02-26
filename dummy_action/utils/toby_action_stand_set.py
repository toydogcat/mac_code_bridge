import tensorflow as tf
import numpy as np
import argparse
import datetime
import json
import math
import time
import cv2
from tools import standardizing as st


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="models/movenet_t.tflite", help="Select the model.")
parser.add_argument("-n", "--name", type=str, default="user", help="The user name.")
parser.add_argument("-i", "--index", type=int, default=0, help="The camera index.")
args = parser.parse_args()

# control the display flag
model_place  = args.model
camera_index = args.index
user_name    = args.name

save_flag    = False
trigger_flag = False
test_flag    = True

# 讀取 user config json file
with open('datas/users_config.json', 'r') as f:
    users_config = json.load(f)


# capture
cap = cv2.VideoCapture(camera_index)
ret, frame = cap.read()
if frame.shape[0] != 720:
    cap.release()
    cap = cv2.VideoCapture(camera_index)

interpreter = tf.lite.Interpreter(model_path=model_place)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def detect_features(keypoints):
    in_flag  = True
    tmp_flag = False

    for keypoint in keypoints:
        if keypoint[2] < 0.4:
            in_flag = False

    if in_flag:
        # right hand
        _al_1 = st.changeAngleLength(keypoints[5], keypoints[7])
        # left hand
        _al_2 = st.changeAngleLength(keypoints[6], keypoints[8])

        if (3 / 16 * np.pi) < -_al_1[0] < (5 / 16 * np.pi) and (11 / 16 * np.pi) < -_al_2[0] < (13 / 16 * np.pi):
            tmp_flag = True
    
    return tmp_flag


if not cap.isOpened():
    print("Cannot open camera")
    exit()

while(True):
    # 擷取影像
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    frame = cv2.flip(frame, 1)
    
    high, width, _ = frame.shape
    # 裁切為正方形
    if high < width:
        x = int((width-high)/2); y = 0
        w = high; h = high; width = high
        frame = frame[y:y+h, x:x+w]
    elif high > width:
        x = 0; y = int((high-width)/2)
        w = width; h = width; high = width
        frame = frame[y:y+h, x:x+w]

    start_model = time.time()

    # thunder 256, light 192
    image = cv2.resize(frame, (input_details[0]['shape'][1], input_details[0]['shape'][1]), interpolation=cv2.INTER_AREA)
    image = tf.expand_dims(image, axis=0)
    interpreter.set_tensor(input_details[0]['index'], tf.cast(image, dtype=tf.uint8).numpy())
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    keypoints_with_scores_np = keypoints_with_scores[0][0]

    end_model = time.time()

    trigger_flag = detect_features(keypoints_with_scores_np)

    if (trigger_flag == True) and (save_flag == False):
        user_base_dictionary = st.changeWholeBodyAngleLengthDictionary(keypoints_with_scores_np)
        user_base_dictionary['keypoints'] = user_base_dictionary['keypoints'].tolist()
        users_config[user_name] = user_base_dictionary
        with open('datas/users_config.json', 'w') as f:
            json.dump(users_config, f)
        
        img_place = 'datas/picture/' + user_name + '.jpg'
        cv2.imwrite(img_place, frame)
        save_flag = True
    
    
    for keypoint in keypoints_with_scores[0][0]:
        y_coordinate = int( keypoint[0] * high  )
        x_coordinate = int( keypoint[1] * width )
        score = keypoint[2]

        if score > 0.8:
            cv2.circle(frame, (x_coordinate, y_coordinate), 2, (255,0,0), 2)
        elif score > 0.4:
            cv2.circle(frame, (x_coordinate, y_coordinate), 2, (255,255,0), 2)
        else:
            cv2.circle(frame, (x_coordinate, y_coordinate), 2, (0,0,255), 2)


    # 白畫布
    white_img = np.ones((720,560,3), dtype=np.uint8) * 255
    
    text = "Time: {time}".format(time = round( (end_model - start_model), 3 ) )
    cv2.putText(white_img, text, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

    text = "Save: {flag}".format( flag = save_flag )
    cv2.putText(white_img, text, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

    text = "Trigger: {flag}".format( flag = trigger_flag )
    cv2.putText(white_img, text, (50, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

    # text = "Test: {flag}".format( flag = test_flag )
    # cv2.putText(white_img, text, (50, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
    
    image_out = cv2.hconcat([white_img, frame])
    # 顯示圖片
    cv2.imshow('live_show', image_out)
    # 按下 q 鍵離開迴圈
    if cv2.waitKey(1) == ord('q'):
        break
    

# 釋放該攝影機裝置
cap.release()
cv2.destroyAllWindows()










