import tensorflow as tf
import datetime
import logging
import cv2


class Toby:
    def __init__(self, model_method="tf"):
        self.model_method = model_method  # tf or tflite
        if model_method == "tf":
            self.model_path = "utils/models/movenet_t/"
        elif model_method == "tflite":
            self.model_path = "utils/models/movenet_t.tflite"
        else:
            self.model_path = "utils/models/movenet_t/"

        self.model = {
            'interpreter' : None,
            'input_details' : None,
            'output_details' : None
        }

        self.image_bgr = None
        self.side = None
        self.keypoints_with_scores = None
        self.cap = None
        self.out_video = None

        # Init
        self.init_log()
        self.init_model()
        self.init_camera()


    def change_model_path(self, path_m='utils/models/movenet.tflite'):
        self.model_path = path_m
        

    def init_model(self, path_m = None):
        logging.info("Init model.")
        if path_m == None:
            path_m = self.model_path
        if self.model_method == 'tflite':
            self.model['interpreter'] = tf.lite.Interpreter(model_path = path_m)
            self.model['interpreter'].allocate_tensors()
            self.model['input_details']  = self.model['interpreter'].get_input_details()
            self.model['output_details'] = self.model['interpreter'].get_output_details()
            self.model['target_height'] = self.model['input_details'][0]['shape'][1]
            self.model['target_width']  = self.model['input_details'][0]['shape'][2]
        elif self.model_method == 'tf':
            model_load = tf.saved_model.load(path_m)
            self.model['model'] = model_load.signatures['serving_default']
            _, self.model['target_height'], self.model['target_width'], _ = self.model['model'].inputs[0].shape


    def model_inference(self, image_bgr = None):
        record_flag = False
        if image_bgr == None:
            image_bgr = self.image_bgr
            record_flag = True
        input_details = self.model['input_details']
        output_details = self.model['output_details']
        image = cv2.resize(image_bgr, (
                        input_details[0]['shape'][1], 
                        input_details[0]['shape'][1]), interpolation=cv2.INTER_AREA)
        image = tf.expand_dims(image, axis=0)
        self.model['interpreter'].set_tensor(input_details[0]['index'], tf.cast(image, dtype=tf.uint8).numpy())
        self.model['interpreter'].invoke()
        keypoints_with_scores = self.model['interpreter'].get_tensor(output_details[0]['index'])
        if record_flag:
            self.keypoints_with_scores = keypoints_with_scores
        else:
            return keypoints_with_scores


    def init_log(self):
        filename = "../utils/logs/" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.log")
        logging.basicConfig(filename=filename, level=logging.INFO)

    def init_camera(self, camera_index=1):
        logging.info("Init camera.")
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            logging.error("Cannot open camera")
            exit()
        fps        = 30
        frame_size = (1280, 720)
        fourcc     = cv2.VideoWriter_fourcc(*'mp4v')
        self.out_video  = cv2.VideoWriter('../record/output.mp4', fourcc, fps, frame_size)
        

    def close_camera(self):
        self.cap.release()
        self.out_video.release()
        cv2.destroyAllWindows()


    def catch_camera(self):
        ret, frame = self.cap.read()
        if not ret:
            logging.error("Can't receive frame (stream end?). Exiting ...")
            self.close_camera()
            exit()

        frame = cv2.flip(frame, 1)
        high, width, _ = frame.shape
        # to square
        if high < width:
            x = int((width-high)/2); y = 0
            w = high; h = high; width = high
            frame = frame[y:y+h, x:x+w]
        elif high > width:
            x = 0; y = int((high-width)/2)
            w = width; h = width; high = width
            frame = frame[y:y+h, x:x+w]

        self.image_bgr = frame



    





