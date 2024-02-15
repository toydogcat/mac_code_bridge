import cv2
import numpy as np
import tensorflow as tf

class Human():
    def __init__(self, user_base_dictionary, rate_method = "real"):
        self.user_base_dictionary = user_base_dictionary
        self.rate = 1.0
        self.rate_method = rate_method
        self.keypoints_with_scores = user_base_dictionary['keypoints']
        self.bounding_boxes = None
        self.nothold_count = 0
        
    def update(self, model_obj, frame):
        if model_obj.model == 'movenet':
            if model_obj.model_type == 'single':
                keypoints_with_scores = model_obj.inference(frame)
                self.keypoints_with_scores = keypoints_with_scores[0][0]
                if self.rate_method == 'real':
                    rate = catch_rate(self.keypoints_with_scores, self.user_base_dictionary['keypoints'])
                    if rate > 0:
                        self.rate = rate
                elif self.rate_method == 'real_1_0_0':
                    self.rate = catch_rate_v_1_0_0(self.keypoints_with_scores, self.user_base_dictionary['keypoints'])
            else:
                multi_keypoints_with_scores = model_obj.inference(frame)
                target_keypoints_with_scores = get_keypoints_with_score(self.keypoints_with_scores, self.bounding_boxes, multi_keypoints_with_scores[0])

                self.keypoints_with_scores = target_keypoints_with_scores[:51].reshape((17,3))  
                _bdd = target_keypoints_with_scores[-5:]
                self.bounding_boxes = {
                    'y_min' : _bdd[0],
                    'x_min' : _bdd[1],
                    'y_max' : _bdd[2],
                    'x_max' : _bdd[3],
                    'score' : _bdd[4],
                }
                if self.rate_method == 'real':
                    rate = catch_rate(self.keypoints_with_scores, self.user_base_dictionary['keypoints'])
                    if rate > 0:
                        self.rate = rate
                elif self.rate_method == 'real_1_0_0':
                    self.rate = catch_rate_v_1_0_0(self.keypoints_with_scores, self.user_base_dictionary['keypoints'])


def get_keypoints_with_score(keypoints_with_scores, bounding_boxes, multi_keypoints_with_scores):
    index_min = 0
    if len(keypoints_with_scores) > 0:
        diff_min = 1000
        for i_k, _keypoints_with_scores in enumerate(multi_keypoints_with_scores):
            diff = 0.0
            _keypoints_with_scores_re = _keypoints_with_scores[:51].reshape((17,3))
            for i in range(17):
                if keypoints_with_scores[i][2] > 0.4 and _keypoints_with_scores_re[i][2] > 0.4:
                    # i_sum += 1
                    diff += abs(keypoints_with_scores[i][0] - _keypoints_with_scores_re[i][0]) \
                            + abs(keypoints_with_scores[i][1] - _keypoints_with_scores_re[i][1])
                else:
                    diff += 0.1
                
                if bounding_boxes == None:
                    diff_denominator = abs(_keypoints_with_scores[-3] - _keypoints_with_scores[-5]) \
                                    + abs(_keypoints_with_scores[-2] - _keypoints_with_scores[-4]) 
                else:
                    diff_denominator = abs(bounding_boxes['y_max'] - bounding_boxes['y_min']) \
                                    + abs(bounding_boxes['x_max'] - bounding_boxes['x_min']) \
                                    + abs(_keypoints_with_scores[-3] - _keypoints_with_scores[-5]) \
                                    + abs(_keypoints_with_scores[-2] - _keypoints_with_scores[-4]) 

                if diff_denominator < 1e-10:
                    diff = 100
                else:
                    diff = diff / ( diff_denominator )

            if diff < diff_min:
                diff_min  = diff
                index_min = i_k
    # print(f"Shape : {multi_keypoints_with_scores.shape}")
    return multi_keypoints_with_scores[index_min].copy()



class Model():
    def __init__(self, model = 'movenet', model_type = 'single'):
        self.model = model
        self.model_type = model_type
        if model == 'movenet':
            if model_type == 'single':
                model_place = 'models/movenet_t.tflite'

                self.interpreter = tf.lite.Interpreter(model_path=model_place)
                self.input_details  = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                self.interpreter.allocate_tensors()
            else:
                model_place = 'models/movenet_multipose_lightning_float16.tflite'
                self.target_size = 256     # 32 倍數

                self.interpreter = tf.lite.Interpreter(model_path=model_place)
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()

                is_dynamic_shape_model = self.input_details[0]['shape_signature'][2] == -1
                if is_dynamic_shape_model:
                    input_tensor_index = self.input_details[0]['index']
                    input_shape = (1, self.target_size, self.target_size, 3)
                    self.interpreter.resize_tensor_input(input_tensor_index, input_shape, strict=True)
                self.interpreter.allocate_tensors()
    
    def inference(self, frame):
        if self.model == 'movenet':
            if self.model_type == 'single':
                # thunder 256, light 192
                image = cv2.resize(frame, (self.input_details[0]['shape'][1], self.input_details[0]['shape'][1]), interpolation=cv2.INTER_AREA)[:,:,::-1]
                image = tf.expand_dims(image, axis=0)
                self.interpreter.set_tensor(self.input_details[0]['index'], tf.cast(image, dtype=tf.uint8).numpy())
                self.interpreter.invoke()
                keypoints_with_scores = self.interpreter.get_tensor(self.output_details[0]['index'])
                return keypoints_with_scores
            else:
                image_target = cv2.resize(frame, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)[:,:,::-1]
                input_tensor = tf.expand_dims(image_target, axis=0)
                self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor.numpy())
                self.interpreter.invoke()
                multi_keypoints_with_scores = self.interpreter.get_tensor(self.output_details[0]['index'])
                return multi_keypoints_with_scores



def distance(pt1, pt2, method = 'xy'):
    if method == 'xy' or method == 'yx':
        return np.sqrt((pt2[1] - pt1[1])**2 + (pt1[0] - pt2[0])**2)
    elif method == 'y':
        return abs(pt1[0] - pt2[0])
    elif method == 'x':
        return abs(pt1[1] - pt2[1])


# face2 come from user config
def catch_rate(face1, face2):   
    face1 = list(map(lambda i: distance(face1[0], face1[i]), [1,2,3,4]))
    face2 = list(map(lambda i: distance(face2[0], face2[i]), [1,2,3,4]))
    rate = [face1[i] / face2[i] for i in range(4)]

    rate_eye = face1[0] / face1[1]
    rate_ear = face1[2] / face1[3]
    
    if 0.8 < rate_eye < 1.2 and 0.8 < rate_ear < 1.2:
        return round(sum(rate)/4, 3)
    else:
        return -1


# old version for catch rate
def catch_rate_v_1_0_0(real_keypoints, user_keypoints):
    threshold = 0.4
    count = 0
    accumulation = 0
    trust_keys = [
        { 'start' : 0,
          'end'   : [5,6,11,12] },
        { 'start' : 5,
          'end'   : [6,11] },
        { 'start' : 6,
          'end'   : [12] },
        { 'start' : 11,
          'end'   : [12] },
    ]
    
    for trust_dictionary in trust_keys:
        for end_index in trust_dictionary['end']:
            i = trust_dictionary['start']
            j = end_index
            if real_keypoints[i][2] > threshold and real_keypoints[j][2] > threshold:
                count += 1
                real_d = np.sqrt(
                        (real_keypoints[i][0] - real_keypoints[j][0]) ** 2 + 
                        (real_keypoints[i][1] - real_keypoints[j][1]) ** 2
                    )
                user_d = np.sqrt(
                        (user_keypoints[i][0] - user_keypoints[j][0]) ** 2 + 
                        (user_keypoints[i][1] - user_keypoints[j][1]) ** 2
                    )
                rate = real_d / user_d
                accumulation += rate
    if count == 0:
        return 1
    rate = accumulation / count
    print(f"rate: {rate}")
    return rate


def distance_map(ch_pt, r_pt):
    _diff = np.array(ch_pt) - np.array(r_pt)
    acc   = 0

    if type(_diff) is np.ndarray:
        for _ in _diff:
            acc += (_**2)
        _d = np.sqrt(acc)
        return _d, _diff

    elif type(_diff) is list:
        for _ in _diff:
            acc += (_**2)
        _d = np.sqrt(acc)
        return _d, _diff

    elif type(_diff) is tuple:
        for _ in _diff:
            acc += (_**2)
        _d = np.sqrt(acc)
        return _d, _diff

    else:
        return abs(_diff), _diff
    return abs(r_pt - ch_pt)


def judgement(_d, real_pt, check_pts, index_begin, real_pre_pt=None, index_range=2, change_map=(lambda x: x) ):
    """
    The output is an array of length 4
    0 : the first element output[0] is a flag, true or false
        true mean pass can go to other index 
        false mean not pass, we should stay here (same index)
    1 : the second element is the score in [0,1] or -1
        if the first element is true
            the second element is the score in [0,1]
        else if the first element is false
            the second element is -1
    2 : the third element is the new index
    3 : the modify array
    """
    _map = map(lambda pt: distance_map(
                                    change_map(pt), 
                                    change_map(real_pt)
                                ), 
                check_pts[index_begin:(index_begin + index_range)])
    distance_obj = np.array(list(_map), dtype=object)
    if len(distance_obj) < index_range:
        _map = map(lambda pt: distance_map(
                                    change_map(pt), 
                                    change_map(real_pt)
                                ), 
                check_pts[:(index_range - len(distance_obj))])
        # more_obj = np.array(list(_map), dtype=object)

        #print('distance_obj : ', distance_obj)
        #print('more_obj : '    , np.array(list(_map), dtype=object))
        # print('len : ', len(distance_obj), ' in : ', index_range)

        # try:
        #     distance_obj = np.append(distance_obj, np.array(list(_map), dtype=object), axis=0)
        # except:
        #     print('distance_obj : ', distance_obj)
        #     print('more_obj : '    , np.array(list(_map), dtype=object))

        distance_obj = np.append(distance_obj, np.array(list(_map), dtype=object), axis=0)

    if distance_obj[0][0] < _d:
        return True, 1, (index_begin+1), distance_obj[0][1]

    # slowly close to target and do nothing
    if real_pre_pt:
        distance_pre_0 = distance_map(change_map(check_pts[0]), change_map(real_pre_pt))
        if distance_pre_0[0] <= distance_obj[0][0] + 0.005:
            return False, -1, index_begin, distance_obj[0][1]

    index_argmin = np.argmin(distance_obj[:,0])
    if index_argmin > 0:
        # print(f"index : {index_begin}, jumping : {index_argmin}")
        return True, (_d/distance_obj[0][0]), ((index_begin+index_argmin) % len(check_pts)), distance_obj[0][1]
    else:
        return False, -1, index_begin, distance_obj[0][1]



def changeAngleList(keypoints_list, k1, k2):
    _f = lambda pt1, pt2: float( np.angle( np.complex(pt2[1] - pt1[1], pt1[0] - pt2[0]) ) )
    return [_f(keypoints[k1], keypoints[k2])  for keypoints in keypoints_list]

def changeAngleLength(pt1, pt2):
    y1 = pt1[0]; x1 = pt1[1]
    y2 = pt2[0]; x2 = pt2[1]
    x_diff =     x2 - x1
    y_diff = - ( y2 - y1 )

    vector = np.complex(x_diff, y_diff)
    # _angle = np.angle(vector, deg=True)
    _angle  = float( np.angle(vector) )
    _length = float( (x_diff ** 2 + y_diff ** 2) ** (0.5) )
    return _angle, _length

def changeVector(pt1, pt2):
    y1 = pt1[0]; x1 = pt1[1]
    y2 = pt2[0]; x2 = pt2[1]
    x_diff = float(     x2 - x1   )
    y_diff = float( - ( y2 - y1 ) )
    return y_diff, x_diff
    
def next_coordinate_from_angle(y, x, _angle, _length):
    x_new = x + np.cos(_angle) * _length
    y_new = y - np.sin(_angle) * _length
    return y_new, x_new

def next_coordinate_from_vector(y, x, y_diff, x_diff):
    x_new = x + x_diff
    y_new = y - y_diff
    return y_new, x_new

# 對於肩膀 髖骨 互算
# 5, 6, 11, 12
def angleLengthTwice(keypoints, twice_list):
    twice_dictionary = {
        'al_13':{},
        'al_24':{},
        'al_31':{},
        'al_42':{},
    }
    index_1, index_2, index_3, index_4 = twice_list

    _keys = changeAngleLength(keypoints[index_1], keypoints[index_3])
    twice_dictionary['al_13']['angle']  = _keys[0]
    twice_dictionary['al_13']['length'] = _keys[1]

    _keys = changeAngleLength(keypoints[index_2], keypoints[index_4])
    twice_dictionary['al_24']['angle']  = _keys[0]
    twice_dictionary['al_24']['length'] = _keys[1]

    _keys = changeAngleLength(keypoints[index_3], keypoints[index_1])
    twice_dictionary['al_31']['angle']  = _keys[0]
    twice_dictionary['al_31']['length'] = _keys[1]

    _keys = changeAngleLength(keypoints[index_4], keypoints[index_2])
    twice_dictionary['al_42']['angle']  = _keys[0]
    twice_dictionary['al_42']['length'] = _keys[1]

    return twice_dictionary


def vectorTwice(keypoints, twice_list):
    twice_dictionary = {
        'al_13':{},
        'al_24':{},
        'al_31':{},
        'al_42':{},
    }
    index_1, index_2, index_3, index_4 = twice_list

    _keys = changeVector(keypoints[index_1], keypoints[index_3])
    twice_dictionary['al_13']['y_diff'] = _keys[0]
    twice_dictionary['al_13']['x_diff'] = _keys[1]

    _keys = changeVector(keypoints[index_2], keypoints[index_4])
    twice_dictionary['al_24']['y_diff'] = _keys[0]
    twice_dictionary['al_24']['x_diff'] = _keys[1]

    _keys = changeVector(keypoints[index_3], keypoints[index_1])
    twice_dictionary['al_31']['y_diff'] = _keys[0]
    twice_dictionary['al_31']['x_diff'] = _keys[1]

    _keys = changeVector(keypoints[index_4], keypoints[index_2])
    twice_dictionary['al_42']['y_diff'] = _keys[0]
    twice_dictionary['al_42']['x_diff'] = _keys[1]

    return twice_dictionary


def angleLengthVectorTwice(keypoints, twice_list):
    twice_dictionary = {
        'al_13':{},
        'al_24':{},
        'al_31':{},
        'al_42':{},
    }
    index_1, index_2, index_3, index_4 = twice_list

    _keys = changeAngleLength(keypoints[index_1], keypoints[index_3])
    twice_dictionary['al_13']['angle']  = _keys[0]
    twice_dictionary['al_13']['length'] = _keys[1]

    _keys = changeAngleLength(keypoints[index_2], keypoints[index_4])
    twice_dictionary['al_24']['angle']  = _keys[0]
    twice_dictionary['al_24']['length'] = _keys[1]

    _keys = changeAngleLength(keypoints[index_3], keypoints[index_1])
    twice_dictionary['al_31']['angle']  = _keys[0]
    twice_dictionary['al_31']['length'] = _keys[1]

    _keys = changeAngleLength(keypoints[index_4], keypoints[index_2])
    twice_dictionary['al_42']['angle']  = _keys[0]
    twice_dictionary['al_42']['length'] = _keys[1]

    _keys = changeVector(keypoints[index_1], keypoints[index_3])
    twice_dictionary['al_13']['y_diff'] = _keys[0]
    twice_dictionary['al_13']['x_diff'] = _keys[1]

    _keys = changeVector(keypoints[index_2], keypoints[index_4])
    twice_dictionary['al_24']['y_diff'] = _keys[0]
    twice_dictionary['al_24']['x_diff'] = _keys[1]

    _keys = changeVector(keypoints[index_3], keypoints[index_1])
    twice_dictionary['al_31']['y_diff'] = _keys[0]
    twice_dictionary['al_31']['x_diff'] = _keys[1]

    _keys = changeVector(keypoints[index_4], keypoints[index_2])
    twice_dictionary['al_42']['y_diff'] = _keys[0]
    twice_dictionary['al_42']['x_diff'] = _keys[1]

    return twice_dictionary




def angleLengthTriple(keypoints, triple_list):
    triple_dictionary = {
        'al_12':{},
        'al_23':{},
        'al_13':{},
    }
    index_1, index_2, index_3 = triple_list

    _keys = changeAngleLength(keypoints[index_1], keypoints[index_2])
    triple_dictionary['al_12']['angle']  = _keys[0]
    triple_dictionary['al_12']['length'] = _keys[1]

    _keys = changeAngleLength(keypoints[index_2], keypoints[index_3])
    triple_dictionary['al_23']['angle']  = _keys[0]
    triple_dictionary['al_23']['length'] = _keys[1]

    _keys = changeAngleLength(keypoints[index_1], keypoints[index_3])
    triple_dictionary['al_13']['angle']  = _keys[0]
    triple_dictionary['al_13']['length'] = _keys[1]

    return triple_dictionary

def vectorTriple(keypoints, triple_list):
    triple_dictionary = {
        'al_12':{},
        'al_23':{},
        'al_13':{},
    }
    index_1, index_2, index_3 = triple_list

    _keys = changeVector(keypoints[index_1], keypoints[index_2])
    triple_dictionary['al_12']['y_diff'] = _keys[0]
    triple_dictionary['al_12']['x_diff'] = _keys[1]

    _keys = changeVector(keypoints[index_2], keypoints[index_3])
    triple_dictionary['al_23']['y_diff'] = _keys[0]
    triple_dictionary['al_23']['x_diff'] = _keys[1]

    _keys = changeVector(keypoints[index_1], keypoints[index_3])
    triple_dictionary['al_13']['y_diff'] = _keys[0]
    triple_dictionary['al_13']['x_diff'] = _keys[1]

    return triple_dictionary

def angleLengthVectorTriple(keypoints, triple_list):
    triple_dictionary = {
        'al_12':{},
        'al_23':{},
        'al_13':{},
    }
    index_1, index_2, index_3 = triple_list

    _keys = changeAngleLength(keypoints[index_1], keypoints[index_2])
    triple_dictionary['al_12']['angle']  = _keys[0]
    triple_dictionary['al_12']['length'] = _keys[1]

    _keys = changeAngleLength(keypoints[index_2], keypoints[index_3])
    triple_dictionary['al_23']['angle']  = _keys[0]
    triple_dictionary['al_23']['length'] = _keys[1]

    _keys = changeAngleLength(keypoints[index_1], keypoints[index_3])
    triple_dictionary['al_13']['angle']  = _keys[0]
    triple_dictionary['al_13']['length'] = _keys[1]

    _keys = changeVector(keypoints[index_1], keypoints[index_2])
    triple_dictionary['al_12']['y_diff'] = _keys[0]
    triple_dictionary['al_12']['x_diff'] = _keys[1]

    _keys = changeVector(keypoints[index_2], keypoints[index_3])
    triple_dictionary['al_23']['y_diff'] = _keys[0]
    triple_dictionary['al_23']['x_diff'] = _keys[1]

    _keys = changeVector(keypoints[index_1], keypoints[index_3])
    triple_dictionary['al_13']['y_diff'] = _keys[0]
    triple_dictionary['al_13']['x_diff'] = _keys[1]

    return triple_dictionary

def changeWholeBodyAngleLengthDictionary(keypoints):
    whole_body_dictionary = {
        'left_arm'  : {},
        'right_arm' : {},
        'left_leg'  : {},
        'right_leg' : {},
        'body'      : {},
        'keypoints' : keypoints,
    }

    whole_body_dictionary['left_arm']  = angleLengthTriple(keypoints, (5,7,9))
    whole_body_dictionary['right_arm'] = angleLengthTriple(keypoints, (6,8,10))
    whole_body_dictionary['left_leg']  = angleLengthTriple(keypoints, (11,13,15))
    whole_body_dictionary['right_leg'] = angleLengthTriple(keypoints, (12,14,16))
    whole_body_dictionary['body']      = angleLengthTwice(keypoints , (5,6,11,12))

    return whole_body_dictionary

def changeWholeBodyVectorDictionary(keypoints):
    whole_body_dictionary = {
        'left_arm'  : {},
        'right_arm' : {},
        'left_leg'  : {},
        'right_leg' : {},
        'body'      : {},
        'keypoints' : keypoints,
    }

    whole_body_dictionary['left_arm']  = vectorTriple(keypoints, (5,7,9))
    whole_body_dictionary['right_arm'] = vectorTriple(keypoints, (6,8,10))
    whole_body_dictionary['left_leg']  = vectorTriple(keypoints, (11,13,15))
    whole_body_dictionary['right_leg'] = vectorTriple(keypoints, (12,14,16))
    whole_body_dictionary['body']      = vectorTwice(keypoints , (5,6,11,12))

    return whole_body_dictionary

def changeWholeBodyAngleLengthVectorDictionary(keypoints):
    whole_body_dictionary = {
        'left_arm'  : {},
        'right_arm' : {},
        'left_leg'  : {},
        'right_leg' : {},
        'body'      : {},
        'keypoints' : keypoints,
    }

    whole_body_dictionary['left_arm']  = angleLengthVectorTriple(keypoints, (5,7,9))
    whole_body_dictionary['right_arm'] = angleLengthVectorTriple(keypoints, (6,8,10))
    whole_body_dictionary['left_leg']  = angleLengthVectorTriple(keypoints, (11,13,15))
    whole_body_dictionary['right_leg'] = angleLengthVectorTriple(keypoints, (12,14,16))
    whole_body_dictionary['body']      = angleLengthVectorTwice(keypoints , (5,6,11,12))

    return whole_body_dictionary

def twoStepPredictCoordinate_vector(_rate, _tutor_label, _start_pt, _part, _words, _keys):
    y_mid, x_mid = next_coordinate_from_vector(
                    _start_pt[0], _start_pt[1], 
                    _tutor_label[_part][_words[0]][_keys[0]] * _rate, 
                    _tutor_label[_part][_words[0]][_keys[1]] * _rate
                )
    return next_coordinate_from_vector(
                    y_mid, x_mid, 
                    _tutor_label[_part][_words[1]][_keys[0]] * _rate, 
                    _tutor_label[_part][_words[1]][_keys[1]] * _rate
                )

def oneStepPredictCoordinate_vector(_rate, _tutor_label, _start_pt, _part, _words, _keys):
    return next_coordinate_from_vector(
                    _start_pt[0], _start_pt[1], 
                    _tutor_label[_part][_words][_keys[0]] * _rate, 
                    _tutor_label[_part][_words][_keys[1]] * _rate
                )

def oneStepAnyPredictCoordinate_vector(_rate, _tutor_label, _start_pt, k1, k2):
    # start_pt = user_keypoints[k1]
    # print(f"k1: {k1}, k2: {k2}")
    _tutor_change = changeVector(_tutor_label['keypoints'][k1], _tutor_label['keypoints'][k2])
    return next_coordinate_from_vector(
                    _start_pt[0], _start_pt[1], 
                    _tutor_change[0] * _rate, 
                    _tutor_change[1] * _rate
                )


def twoStepPredictCoordinate_angle(_rate, _user_base, _tutor_label, _start_pt, _part, _words, _keys):
    y_mid, x_mid = next_coordinate_from_angle(
                    _start_pt[0], _start_pt[1], 
                    _tutor_label[_part][_words[0]][_keys[0]], 
                    _user_base[_part][_words[0]][_keys[1]] * _rate
                )
    return next_coordinate_from_angle(
                    y_mid, x_mid, 
                    _tutor_label[_part][_words[1]][_keys[0]], 
                    _user_base[_part][_words[1]][_keys[1]] * _rate
                )

def oneStepPredictCoordinate_angle(_rate, _user_base, _tutor_label, _start_pt, _part, _words, _keys):
    return next_coordinate_from_angle(
                    _start_pt[0], _start_pt[1],
                    _tutor_label[_part][_words][_keys[0]],
                    _user_base[_part][_words][_keys[1]] * _rate
                )

def oneStepAnyPredictCoordinate_angle(_rate, _tutor_label, _start_pt, k1, k2):
    # start_pt = user_keypoints[k1]
    _keys = changeAngleLength(_tutor_label['keypoints'][k1], _tutor_label['keypoints'][k2])
    return next_coordinate_from_angle(
                    _start_pt[0], _start_pt[1], 
                    _keys[0], 
                    _keys[1] * _rate
                )




'''
Available Cases : 

Case : k1, k2
    i_part = body, left_arm, right_arm, left_leg, right_leg
    i_type = vector or angle
    i_method = 0
    k1 = the first index in keypoints
    k2 = the second index in keypoints

    meaningless : i_port

Case : four limbs
    i_part = left_arm, right_arm, left_leg, right_leg
    i_type = vector or angle
    i_method = 12, 23, 13, 123

    meaningless : k1, k2

Case : body
    i_part = body
    i_type = vector or angle
    i_method = 13, 31, 24, 42

    meaningless : k1, k2

Case : slope
    i_type = slope
    k1 = the first index in keypoints
    k2 = the second index in keypoints

    meaningless : i_part, i_method

Case : keypoints
    i_part = x, y, xy, yx
    i_type = keypoints
    k1 = the first index in user keypoints for user
    k2 = the second index in user keypoints for user
    k3 = the first index in user keypoints for tutor
    k4 = the second index in user keypoints for tutor

    meaningless : i_method

'''
# 要改
def got_user_coordinate_from_both(user_base, tutor_label, user_obj, 
                                i_type, i_part, i_method,
                                k1 = 0, k2 = 1, k3 = 2, k4 = 3):
    if user_obj.rate_method == "real":
        rate = user_obj.rate
        if rate < 0:
            rate = 1
    elif user_obj.rate_method == "real_1_0_0":
        rate = user_obj.rate
    else:
        rate = 1

    user_keypoints = user_obj.keypoints_with_scores
        
    if i_type == 'vector':
        if i_method == 123:
            if i_part == 'left_arm':
                start_pt = user_keypoints[5]
            elif i_part == 'right_arm':
                start_pt = user_keypoints[6]
            elif i_part == 'left_leg':
                start_pt = user_keypoints[11]
            elif i_part == 'right_leg':
                start_pt = user_keypoints[12]
            
            words = ('al_12', 'al_23')
            keys = ('y_diff', 'x_diff')
            return twoStepPredictCoordinate_vector(
                                _rate = rate, 
                                _tutor_label = tutor_label, 
                                _start_pt = start_pt, 
                                _part = i_part, 
                                _words = words, 
                                _keys = keys
                    )
        elif i_method == 13:
            if i_part == 'left_arm':
                start_pt = user_keypoints[5]
            elif i_part == 'right_arm':
                start_pt = user_keypoints[6]
            elif i_part == 'left_leg':
                start_pt = user_keypoints[11]
            elif i_part == 'right_leg':
                start_pt = user_keypoints[12]
            elif i_part == 'body':
                start_pt = user_keypoints[5]

            words = 'al_13'
            keys = ('y_diff', 'x_diff')
            return oneStepPredictCoordinate_vector(
                                _rate = rate,
                                _tutor_label = tutor_label,
                                _start_pt = start_pt,
                                _part = i_part,
                                _words = words,
                                _keys = keys
                    )
        elif i_method == 12:
            if i_part == 'left_arm':
                start_pt = user_keypoints[5]
            elif i_part == 'right_arm':
                start_pt = user_keypoints[6]
            elif i_part == 'left_leg':
                start_pt = user_keypoints[11]
            elif i_part == 'right_leg':
                start_pt = user_keypoints[12]

            words = 'al_12'
            keys = ('y_diff', 'x_diff')
            return oneStepPredictCoordinate_vector(
                                _rate = rate,
                                _tutor_label = tutor_label, 
                                _start_pt = start_pt, 
                                _part = i_part, 
                                _words = words, 
                                _keys = keys
                    )
        elif i_method == 23:
            if i_part == 'left_arm':
                start_pt = user_keypoints[7]
            elif i_part == 'right_arm':
                start_pt = user_keypoints[8]
            elif i_part == 'left_leg':
                start_pt = user_keypoints[13]
            elif i_part == 'right_leg':
                start_pt = user_keypoints[14]

            words = 'al_23'
            keys = ('y_diff', 'x_diff')
            return oneStepPredictCoordinate_vector(
                                _rate = rate,
                                _tutor_label = tutor_label, 
                                _start_pt = start_pt, 
                                _part = i_part, 
                                _words = words, 
                                _keys = keys
                    )
        elif i_method == 24:
            if i_part == 'body':
                start_pt = user_keypoints[6]
            else:
                start_pt = user_keypoints[6]

            words = 'al_24'
            keys = ('y_diff', 'x_diff')
            return oneStepPredictCoordinate_vector(
                                _rate = rate,
                                _tutor_label = tutor_label,
                                _start_pt = start_pt,
                                _part = i_part,
                                _words = words,
                                _keys = keys
                    )
        elif i_method == 31:
            if i_part == 'body':
                start_pt = user_keypoints[11]
            else:
                start_pt = user_keypoints[11]

            words = 'al_31'
            keys = ('y_diff', 'x_diff')
            return oneStepPredictCoordinate_vector(
                                _rate = rate,
                                _tutor_label = tutor_label,
                                _start_pt = start_pt,
                                _part = i_part,
                                _words = words,
                                _keys = keys
                    )
        elif i_method == 42:
            if i_part == 'body':
                start_pt = user_keypoints[12]
            else:
                start_pt = user_keypoints[12]

            words = 'al_42'
            keys = ('y_diff', 'x_diff')
            return oneStepPredictCoordinate_vector(
                                _rate = rate,
                                _tutor_label = tutor_label,
                                _start_pt = start_pt,
                                _part = i_part,
                                _words = words,
                                _keys = keys
                    )
        elif i_method == 0:
            start_pt = user_keypoints[k1]
            return oneStepAnyPredictCoordinate_vector(_rate = rate, 
                                _tutor_label = tutor_label, 
                                _start_pt = start_pt, 
                                k1 = k1, 
                                k2 = k2)
    elif i_type == 'angle':
        if i_method == 123:
            if i_part == 'left_arm':
                start_pt = user_keypoints[5]
            elif i_part == 'right_arm':
                start_pt = user_keypoints[6]
            elif i_part == 'left_leg':
                start_pt = user_keypoints[11]
            elif i_part == 'right_leg':
                start_pt = user_keypoints[12]

            words = ('al_12', 'al_23')
            keys = ('angle', 'length')
            return twoStepPredictCoordinate_angle(
                                _rate = rate,
                                _user_base = user_base,
                                _tutor_label = tutor_label, 
                                _start_pt = start_pt, 
                                _part = i_part, 
                                _words = words, 
                                _keys = keys
                    )
        elif i_method == 13:
            if i_part == 'left_arm':
                start_pt = user_keypoints[5]
            elif i_part == 'right_arm':
                start_pt = user_keypoints[6]
            elif i_part == 'left_leg':
                start_pt = user_keypoints[11]
            elif i_part == 'right_leg':
                start_pt = user_keypoints[12]
            elif i_part == 'body':
                start_pt = user_keypoints[5]

            words = 'al_13'
            keys = ('angle', 'length')
            return oneStepPredictCoordinate_angle(
                                _rate = rate,
                                _user_base = user_base,
                                _tutor_label = tutor_label, 
                                _start_pt = start_pt, 
                                _part = i_part, 
                                _words = words, 
                                _keys = keys
                    )
        elif i_method == 12:
            if i_part == 'left_arm':
                start_pt = user_keypoints[5]
            elif i_part == 'right_arm':
                start_pt = user_keypoints[6]
            elif i_part == 'left_leg':
                start_pt = user_keypoints[11]
            elif i_part == 'right_leg':
                start_pt = user_keypoints[12]

            words = 'al_12'
            keys = ('angle', 'length')
            return oneStepPredictCoordinate_angle(
                                _rate = rate,
                                _user_base = user_base,
                                _tutor_label = tutor_label, 
                                _start_pt = start_pt, 
                                _part = i_part, 
                                _words = words, 
                                _keys = keys
                    )
        elif i_method == 23:
            if i_part == 'left_arm':
                start_pt = user_keypoints[7]
            elif i_part == 'right_arm':
                start_pt = user_keypoints[8]
            elif i_part == 'left_leg':
                start_pt = user_keypoints[13]
            elif i_part == 'right_leg':
                start_pt = user_keypoints[14]
                
            words = 'al_23'
            keys = ('angle', 'length')
            return oneStepPredictCoordinate_angle(
                                _rate = rate,
                                _user_base = user_base,
                                _tutor_label = tutor_label,
                                _start_pt = start_pt,
                                _part = i_part,
                                _words = words,
                                _keys = keys
                    )
        elif i_method == 24:
            if i_part == 'body':
                start_pt = user_keypoints[6]
            else:
                start_pt = user_keypoints[6]
                
            words = 'al_24'
            keys = ('angle', 'length')
            return oneStepPredictCoordinate_angle(
                                _rate = rate,
                                _user_base = user_base,
                                _tutor_label = tutor_label,
                                _start_pt = start_pt,
                                _part = i_part,
                                _words = words,
                                _keys = keys
                    )
        elif i_method == 31:
            if i_part == 'body':
                start_pt = user_keypoints[11]
            else:
                start_pt = user_keypoints[11]
                
            words = 'al_31'
            keys = ('angle', 'length')
            return oneStepPredictCoordinate_angle(
                                _rate = rate,
                                _user_base = user_base,
                                _tutor_label = tutor_label,
                                _start_pt = start_pt,
                                _part = i_part,
                                _words = words,
                                _keys = keys
                    )
        elif i_method == 42:
            if i_part == 'body':
                start_pt = user_keypoints[12]
            else:
                start_pt = user_keypoints[12]
                
            words = 'al_42'
            keys = ('angle', 'length')
            return oneStepPredictCoordinate_angle(
                                _rate = rate,
                                _user_base = user_base,
                                _tutor_label = tutor_label,
                                _start_pt = start_pt,
                                _part = i_part,
                                _words = words,
                                _keys = keys
                    )
        elif i_method == 0:
            start_pt = user_keypoints[k1]
            return oneStepAnyPredictCoordinate_angle(_rate = rate, 
                                _tutor_label = tutor_label, 
                                _start_pt = start_pt, 
                                k1 = k1, 
                                k2 = k2)
    elif i_type == 'slope':
        tutor_keypoints = tutor_label['keypoints']

        tutor_angle = changeAngleLength(tutor_keypoints[k1], tutor_keypoints[k2])[0]
        user_angle  = changeAngleLength(user_keypoints[k1] , user_keypoints[k2] )[0]

        return tutor_angle, user_angle

    elif i_type == 'keypoints':
        
        user_distance  = distance(user_keypoints[k1], user_keypoints[k2], method=i_part)
        tutor_distance = distance(user_keypoints[k3], user_keypoints[k4], method=i_part)

        return tutor_distance, user_distance











