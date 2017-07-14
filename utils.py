import sys, thread, time, os
from config import *
sys.path.insert(0, leap_lib)
sys.path.insert(0, more_leap)
import Leap, ctypes

def read_frame(filename):

    Leap.Controller()
    new_frame = Leap.Frame()
    with open(os.path.realpath(filename), 'rb') as data_file:
        data = data_file.read()

    leap_byte_array = Leap.byte_array(len(data))
    address = leap_byte_array.cast().__long__()
    ctypes.memmove(address, data, len(data))
    new_frame.deserialize((leap_byte_array, len(data)))
    return new_frame

def get_normalised_fingers_features(hand):
    hand_x_basis = hand.basis.x_basis
    hand_y_basis = hand.basis.y_basis
    hand_z_basis = hand.basis.z_basis
    hand_origin = hand.palm_position
    hand_transform = Leap.Matrix(hand_x_basis, hand_y_basis, hand_z_basis, hand_origin)
    hand_transform = hand_transform.rigid_inverse()

    features = []
    for finger in hand.fingers:
#        transformed_position = hand_transform.transform_point(finger.tip_position)
#        transformed_direction = hand_transform.transform_direction(finger.direction) 
        transformed_position = finger.tip_position
        transformed_direction = finger.direction
        features.append(transformed_position.x)
        features.append(transformed_position.y)
        features.append(transformed_position.z)
        features.append(transformed_direction.x)
        features.append(transformed_direction.y)
        features.append(transformed_direction.z)
    return features
