import sys, os
from config import leap_lib, more_leap
sys.path.insert(0, leap_lib)
sys.path.insert(0, more_leap)
import Leap, ctypes, itertools
import matplotlib.pyplot as plt
import numpy as np


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

def is_primitive(thing):
    return type(thing) in (int, float)

valid_features = ('center', 'next_joint', 'prev_joint', 'hands', 'fingers', 'arm', 'basis', 'x_basis', 'y_basis', 'origin', 'z_basis', 'x', 'y', 'z', 'pitch', 'roll', 'yaw', 'confidence', 'direction', 'grab_strength', 'palm_normal', 'palm_position', 'palm_velocity', 'pinch_strength', 'sphere_center', 'sphere_radius', 'stabilized_palm_position', 'stabilized_tip_position', 'tip_position', 'tip_velocity', 'wrist_position')


def flatten(obj):
    l = [val for val in dir(obj) if val in valid_features]
    for el in l:
        for sub in flatten(getattr(obj, el)):
            yield sub
    if isinstance(obj, float):
        yield obj

def get_features(hand):
    return itertools.chain(flatten(hand), get_normalised_fingers_features(hand))
    
def get_hand_features(hand):
    return flatten(hand)

def get_finger_features(hand):
    return get_normalised_fingers_features(hand)

def get_normalised_fingers_features(hand):
    hand_x_basis = hand.basis.x_basis
    hand_y_basis = hand.basis.y_basis
    hand_z_basis = hand.basis.z_basis
    hand_origin = hand.palm_position
    hand_transform = Leap.Matrix(hand_x_basis, hand_y_basis, hand_z_basis, hand_origin)
    hand_transform = hand_transform.rigid_inverse()

    features = []
    # force some sort of order for ML
    for finger in hand.fingers:
        if finger.type == 0:
            thumb = process_finger(finger, hand_transform)
        elif finger.type == 1:
            index = process_finger(finger, hand_transform)
        elif finger.type == 2:
            middle = process_finger(finger, hand_transform)
        elif finger.type == 3:
            ring = process_finger(finger, hand_transform)
        elif finger.type == 4:
            pinky = process_finger(finger, hand_transform)

    features = itertools.chain(thumb, index, middle, ring, pinky)

    return features

def process_finger(finger, hand_transform):
        # flatten finger with pointable features
        features = flatten(finger)
        for i in range(4):
            features = itertools.chain(features, flatten(finger.bone(i)))

        transformed_position = hand_transform.transform_point(finger.tip_position)
        transformed_direction = hand_transform.transform_direction(finger.direction) 
#        transformed_position = finger.tip_position
#        transformed_direction = finger.direction
        features = itertools.chain(features, flatten(transformed_position), flatten(transformed_direction))
        
        return features

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')