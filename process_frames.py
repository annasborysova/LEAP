# -*- coding: utf-8 -*-
import utils, pickle, glob, os

def process_frame(frame, feature_set_type='all'):
    if len(frame.hands) != 1:
#        print("Bad frame: Incorrect number of hands " + str(len(frame.hands)))
        return []
    if feature_set_type == 'all':
        return utils.get_features(frame.hands[0])
    if feature_set_type == 'fingers_only':
        return utils.get_finger_features(frame.hands[0])
    if feature_set_type == 'hands_only':
        return utils.get_hand_features(frame.hands[0])


def rewrite(path, feature_set_type):
    for foldername in sorted(glob.glob(os.path.join(path, "Leap_*"))):
        for filename in glob.glob(os.path.join(foldername, "*.txt")):
            frame_features = process_frame(utils.read_frame(filename), feature_set_type)
            with open(filename[:-4] + '_' + feature_set_type + ".features", 'wb') as fp:
                pickle.dump([x for x in frame_features], fp)
            
def rewrite_all():
    paths = ["Leap_Data", os.path.join("Leap_Data", "DataGath2"), os.path.join("Leap_Data", "DataGath3")]
    feature_set_types = ['fingers_only', 'hands_only', 'all']
    for path in paths:
        for feature_set_type in feature_set_types:
            print("rewriting in " + path + " for feature_set_type " + feature_set_type )
            rewrite(path, feature_set_type)

if __name__=="__main__":
#    rewrite("Leap_Data\\")
#    rewrite("Leap_Data\\DataGath2\\")
#    rewrite("Leap_Data\\DataGath3\\")
    rewrite_all()
    import winsound
    winsound.Beep(300,2000)
