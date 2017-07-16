import glob, utils, string
import sklearn

print sklearn.__version__


def read_data(path):
    gesture_names = []
    gesture_data = []
#    gestures_done = 0
    for foldername in sorted(glob.glob(path + 'Leap_*')):
#        if gestures_done > 50:
#            break
#        gestures_done += 1
        letter_index = foldername[len(path)+5]
        gesture = []
#        gesture_names.append(letter_index)
        
        ideal_frame_number = 1
        frame_number = 0
        
        #for every frame in a gesture
        for filename in glob.glob(foldername + '\\*'):
            # get features from frame
            frame_features = process_frame(utils.read_frame(filename))
            
            # if there's data/features in frame
            if frame_features:
                frame_number += 1
                gesture.extend(frame_features)
#                gesture_data.append(frame_features)
#                gesture_names.append(letter_index)
                
                # only add features from 5 frames
                if frame_number == ideal_frame_number:
                    break
            else:
                # bad frame
                print filename
        
        # only consider a gesture if correct number of frames/features
        if frame_number == ideal_frame_number:
            gesture_data.append(gesture)
            gesture_names.append(letter_index)
        else:
            print "Rejected gesture: " + foldername
            
            
    return gesture_names, gesture_data

def process_frame(frame):
    if len(frame.hands) != 1:
        print("Bad frame: Incorrect number of hands " + str(len(frame.hands)))
        return False
    else:
#        for attribute in dir(frame.hands[0]):
##            print attribute
#            if callable(getattr(frame.hands[0], attribute)):
#                print "callable: " + attribute
#            else:
#                print "not calllable: " + attribute
        print "good frame"
#        exit()
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