import numpy as np



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


def catch_rate(user_keypoints, real_keypoints):
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
    return rate



'''
Available Cases : 

Case : four limbs
    i_port = left_arm, right_arm, left_leg, right_leg
    i_type = vector or angle
    i_method = 12, 23, 13, 123

    meaningless : k1, k1

Case : body
    i_port = body
    i_type = vector or angle
    i_method = 13, 31, 24, 42

    meaningless : k1, k1

Case : slope
    i_type = slope
    k1 = the first index in keypoints
    k2 = the second index in keypoints

    meaningless : i_port, i_method

'''
def got_user_coordinate_from_both(user_base, tutor_label, user_keypoints, 
                                i_type, i_part, i_method, 
                                rate_method = "real",
                                k1 = 0, k2 = 1):
    if rate_method == "real":
        rate = catch_rate(
            user_keypoints = user_base['keypoints'], 
            real_keypoints = user_keypoints
        )
    else:
        rate = 1
        
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
    elif i_type == 'slope':
        tutor_keypoints = tutor_label['keypoints']

        tutor_angle = changeAngleLength(tutor_keypoints[k1], tutor_keypoints[k2])[0]
        user_angle  = changeAngleLength(user_keypoints[k1] , user_keypoints[k2] )[0]

        return tutor_angle, user_angle










