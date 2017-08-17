from sklearn import svm, neighbors, tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
#from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.model_selection import GridSearchCV
import os
from read_data import read_data

print("start")
frames_per_gesture=10
separate_frames=True
feature_set_type="fingers_only"
gesture_data2, gesture_names2 = read_data(os.path.join("Leap_Data", "DataGath3"), frames_per_gesture, separate_frames, feature_set_type)
gesture_data, gesture_names = read_data(os.path.join("Leap_Data"), frames_per_gesture, separate_frames, feature_set_type)

gesture_names.extend(gesture_names2)
gesture_data.extend(gesture_data2)

gesture_data2, gesture_names2 = read_data(os.path.join("Leap_Data", "DataGath2"), frames_per_gesture, separate_frames, feature_set_type)

#gesture_names.extend(gesture_names2)
#gesture_data.extend(gesture_data2)


#training_data, test_data, training_target, test_target = train_test_split(process.gesture_data, process.gesture_names, test_size=0.25, random_state=0)
training_data, training_target = gesture_data, gesture_names
test_data, test_target = gesture_data2, gesture_names2
assert len(training_data) == len(training_target)
assert len(test_data) == len(test_target)
print("I'm alive")
#selector = SelectKBest(k=20)
selector = VarianceThreshold(threshold=(.8 * (1 - .8)))
training_data = selector.fit_transform(training_data, training_target)
test_data = selector.transform(test_data)

classifiers = {        
        'SVC': svm.SVC(),
        'SVCP': svm.SVC(gamma=0.001, C=10),
        'SVCR': svm.SVC(gamma=0.0001, C=500),
        'NB ': GaussianNB(),
        'BNB': BernoulliNB(),
        'NBU': neighbors.KNeighborsClassifier(5, weights='uniform'),
        'NBD': neighbors.KNeighborsClassifier(5, weights='distance'),
        'TRE': tree.DecisionTreeClassifier(),
        'GBC': GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0),
        'RFC': RandomForestClassifier(),
        'MLP': MLPClassifier(),
    }


parameters = {'kernel':('linear', 'rbf'), 'C':[1, 500]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(training_data, training_target)
print(clf.score(test_data, test_target))

scores = [(n, clf.fit(training_data, training_target).score(test_data,
    test_target)) for n, clf in classifiers.iteritems()]

for name, score in sorted(scores, key=lambda t: t[1], reverse=True):
    print name, score

#for n, clf in classifiers.iteritems():
#    print "CLASSIFIER: " + n
#    gesture_pred = clf.fit(training_data, training_target).predict(test_data)
#    target_names = list(string.ascii_lowercase)
#    cm = confusion_matrix(test_target, gesture_pred, target_names)
#    utils.plot_confusion_matrix(cm, classes=target_names, title=n)


#    print(classification_report(test_target, gesture_pred, target_names=target_names))

import winsound
winsound.Beep(500,500)
winsound.Beep(500,500)
