import glob, pickle, os
import string

def read_data(path, frames_per_gesture=10, separate_frames=True, feature_set_type="all"):
    gesture_names = []
    gesture_data = []

    for foldername in sorted(glob.glob(os.path.join(path, "Leap_*"))):
        
        print(foldername)
        gesture = process_gesture_folder(foldername, frames_per_gesture, separate_frames, feature_set_type)
        
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

def process_gesture_folder(foldername, frames_per_gesture, separate_frames, feature_set_type):
        gesture = []
        frame_num = 0
        
        # for every frame in a gesture
        for filename in glob.glob(os.path.join(foldername, "*.txt")):
            
            # get features from frame
            with open(filename[:-4] + '_' + feature_set_type + ".features", 'rb') as fp:
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
