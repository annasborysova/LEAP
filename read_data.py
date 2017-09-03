import glob, pickle, os, sys
import string, utils, process_frames
from operator import add


def read_data(path, frames_per_gesture=1, separate_frames=False, feature_set_type="all", average=False):
    gesture_names = []
    gesture_data = []
    if average:
        separate_frames = False

    for foldername in sorted(glob.glob(os.path.join(path, "Leap_*"))):
        
        
        gesture = process_gesture_folder(foldername, frames_per_gesture, separate_frames, feature_set_type, average)

        # only consider a gesture if correct number of frames/features            
        if gesture:
#            print gesture
            letter = foldername[len(path)+6]
#            letter = string.ascii_lowercase.index(foldername[len(path)+6])
            
            # gesture_data is a list of lists
            # gesture_names is a list of letters
            if separate_frames:
                # gesture is a list of lists
                gesture_data.extend(gesture)
                gesture_names.extend([letter] * len(gesture))
            else:
                # gesture is a list of floats
                gesture_data.append(gesture)
                gesture_names.append(letter)
                
    return gesture_data, gesture_names


def process_gesture_folder(foldername, frames_per_gesture, separate_frames, feature_set_type, average):
        gesture = []
        frame_num = 0
        
        if average:
            return get_average(foldername, feature_set_type)
        
        # for every frame in a gesture
        for filename in glob.glob(os.path.join(foldername, "*" + feature_set_type + ".features")):
        

#             get features from frame
            with open(filename, 'rb') as fp:
                frame_features = pickle.load(fp)

            # if frame_features empty, nothing added to gesture
            if frame_features:
                gesture.append([x for x in frame_features]) if separate_frames else gesture.extend(frame_features)
                frame_num += 1
            
            # only add features from required number of frames
            if frame_num == frames_per_gesture:
                return gesture
#        print "not enough frames to return gesture"
#        return False
        if separate_frames:
            return gesture
        else:
            return []

def get_feature_names(path, feature_set_type):
    with open(os.path.join(path, feature_set_type + ".feature_names"), 'rb') as fp:
        return pickle.load(fp)
    

def get_average(path, feature_set_type):
    for foldername in sorted(glob.glob(os.path.join(path, "Leap_*"))):
        features = []
        count = 0
        for filename in glob.glob(os.path.join(foldername, "*" + feature_set_type + ".features")):
            with open(filename, 'rb') as fp:
                new_features= pickle.load(fp)
                if not new_features:
                    continue
                if features:
                    count += 1
                    map(add, features, new_features)
                else:
                    count += 1
                    features = new_features
        
        return [x/(count) for x in features]  