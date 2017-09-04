# -*- coding: utf-8 -*-
import utils, pickle, glob, os


def process_frame(frame, feature_set_type='all', labels=False):
    if len(frame.hands) != 1:
#        print("Bad frame: Incorrect number of hands " + str(len(frame.hands)))
        return []
    if feature_set_type == 'all':
        return utils.get_features(frame.hands[0], labels=labels)    
    if feature_set_type == 'fingers_only':
        return utils.get_finger_features(frame.hands[0], labels=labels)
    if feature_set_type == 'hands_only':
        return utils.get_hand_features(frame.hands[0], labels=labels)


def rewrite(path, feature_set_type):
    feature_names = []
    for gesture_count, foldername in enumerate(sorted(glob.glob(os.path.join(path, "Leap_*")))):
        print(foldername)
        for count, filename in enumerate(glob.glob(os.path.join(foldername, "*.data"))):
            if count%5 != 0:
                continue
            if gesture_count == 0 and count == 0:
                print("getting feature names")
                frame_features = process_frame(utils.read_frame(filename), feature_set_type, labels=True)
                frame_features = zip(*frame_features)
                feature_names = frame_features[1]
                frame_features = frame_features[0]
                with open(os.path.join(path, feature_set_type + ".feature_names"), 'wb') as fp:
                    print(os.path.join(path, feature_set_type + ".feature_names"))
                    pickle.dump([x for x in feature_names], fp)
                    print("written")

            frame_features = process_frame(utils.read_frame(filename), feature_set_type)
            if frame_features:
                with open(filename[:-5] + '_' + feature_set_type + ".features", 'wb') as fp:
                    pickle.dump([x for x in frame_features], fp)
                    
            
def rewrite_all():
#    paths = ["Leap_Data", os.path.join("Leap_Data", "DataGath2"), os.path.join("Leap_Data", "DataGath3"), os.path.join("Leap_Data", "Participant 0")]
#    paths = [os.path.join("Leap_Data", "Participant 0")]
    paths = [os.path.join("Leap_Data", "Legit_Data", "Participant " + str(x), "Leap") for x in range(0, 3)]
#    paths = [os.path.join("Leap_Data", "Legit_Data", "Participant 12", "Leap")]

#    feature_set_types = ['fingers_only', 'hands_only', 'all']
    feature_set_types = ['all']
    for path in paths:
        for feature_set_type in feature_set_types:
            print("rewriting in " + path + " for feature_set_type " + feature_set_type )
            rewrite(path, feature_set_type)


if __name__=="__main__":
#    rewrite(os.path.join("Leap_Data", "Participant 0"))
#    rewrite("Leap_Data\\DataGath2\\")
#    rewrite("Leap_Data\\DataGath3\\")
    rewrite_all()
    import winsound
    winsound.Beep(300,2000)
