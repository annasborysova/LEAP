import glob, utils, string
import sklearn

def read_data(path, max_gestures=float("inf"), frames_per_gesture=1, separate_frames=True):
    gesture_names = []
    gesture_data = []

    for foldername in sorted(glob.glob(path + 'Leap_*')):
        if len(gesture_names) > max_gestures:
            break
                
        gesture = process_gesture_folder(foldername, frames_per_gesture, separate_frames)
        
        # only consider a gesture if correct number of frames/features            
        if gesture:
            letter_index = foldername[len(path)+5]
            if separate_frames:
                gesture_data.extend(gesture)
                gesture_names.extend([letter_index] * len(gesture))
            else:
                gesture_data.append(gesture)
                gesture_data.append(letter_index)
    print gesture_data
    print gesture_names
    return gesture_names, gesture_data

def process_gesture_folder(foldername, frames_per_gesture, separate_frames):
        gesture = []
        frame_num = 0
        #for every frame in a gesture
        for filename in glob.glob(foldername + '\\*'):
            # get features from frame
            frame_features = process_frame(utils.read_frame(filename))
            
            # if frame_features empty, nothing added to gesture
            if frame_features:
                gesture.append([x for x in frame_features]) if separate_frames else gesture.extend(frame_features)
                frame_num += 1
            
            # only add features from required number of frames
            if frame_num == frames_per_gesture:
                return gesture

def process_frame(frame):
    if len(frame.hands) != 1:
#        print("Bad frame: Incorrect number of hands " + str(len(frame.hands)))
        return False
    return utils.get_features(frame.hands[0])

gesture_names2, gesture_data2 = read_data("..\\HonoursProject\\DataGathering\\DataGath2\\")
gesture_names, gesture_data = read_data("..\\HonoursProject\\DataGathering\\")

gesture_names.extend(gesture_names2)
gesture_data.extend(gesture_data2)

#
#clf = svm.LinearSVC()
#train_classifier(clf, "..\\HonoursProject\\DataGathering\\")
#
#gesture = process_frame(utils.read_frame("..\\HonoursProject\\DataGathering\\Leap_b_1499275525.96\\1499275527.39.txt"))
#print clf.predict(gesture)