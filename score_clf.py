from sklearn import svm, neighbors, tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import os, string, utils
from read_data import read_data


frames_per_gesture=2
separate_frames=False
feature_set_type="all"
train_paths = ["Leap_Data", os.path.join("Leap_Data", "DataGath3")]
test_paths = [os.path.join("Leap_Data", "DataGath2")]

#training_data = []
#training_target = []
#for path in train_paths:
#    data, target = read_data()
#    training_data.extend
    
gesture_data2, gesture_names2 = read_data(os.path.join("Leap_Data", "DataGath3"), frames_per_gesture, separate_frames, feature_set_type)
gesture_data, gesture_names = read_data(os.path.join("Leap_Data"), frames_per_gesture, separate_frames, feature_set_type)

gesture_names.extend(gesture_names2)
gesture_data.extend(gesture_data2)

gesture_data2, gesture_names2 = read_data(os.path.join("Leap_Data", "DataGath2"), frames_per_gesture, separate_frames, feature_set_type)

gesture_names.extend(gesture_names2)
gesture_data.extend(gesture_data2)


training_data, test_data, training_target, test_target = train_test_split(gesture_data, gesture_names, test_size=0.25, random_state=0)
#training_data, training_target = gesture_data, gesture_names
#test_data, test_target = gesture_data2, gesture_names2


selector = SelectKBest(k=500)
#selector = VarianceThreshold(threshold=(.8 * (1 - .8)))
training_data = selector.fit_transform(training_data, training_target)
test_data = selector.transform(test_data)



classifiers = {        
        'SVC': (svm.SVC(), 
                {
                    'kernel':('linear', 'rbf',  'poly', 'sigmoid'),
                    'C':range(1,500),
                    'gamma':[1, 0.1, 0.01, 0.001, 0.0001],
                    'degree': range(10),
                    'coef0': range(0,500,10),
                    'probability': [True, False],
                    'shrinking': [True, False],
#                    'tol': [1.0/(10**x) for x in range(50)],
                    'class_weight': ['balanced', None],
                    'decision_function_shape': ['ovo', 'ovr'],
                }),
        'BNB': (BernoulliNB(),
                {
                    'alpha': [1.0/(10**x) for x in range(50)],
                    'fit_prior': [True, False],                    
                }),
        'kNN': (neighbors.KNeighborsClassifier(),
                {
                    'n_neighbors': range(1,10),
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    'leaf_size': range(1, 100, 10),
                    'p': range(1,10),
                }),
        'MLP': (MLPClassifier(),
                {
#                        'beta_1': [9.0/(10**x) for x in range(50)], 
                    'warm_start': [True, False],
#                        'beta_2': [9.0/(10**x) for x in range(50)],
                    'shuffle': [True, False],
                    'verbose': [True, False],
                    'nesterovs_momentum': [True, False], 
#                        'hidden_layer_sizes': [(100,),], # add more values
#                        'epsilon': 1e-08, 
                    'activation': ('identity', 'logistic', 'tanh', 'relu'), 
#                        'batch_size': 'auto', 
#                        'power_t': 0.5, 
#                        'random_state': None, 
                    'learning_rate_init': [1.0/(10**x) for x in range(10)], 
                    'tol': [1.0/(10**x) for x in range(50)], # likes to be 50/high
                    'validation_fraction': [1.0/(10**x) for x in range(1,10)], 
                    'alpha': [1.0/(10**x) for x in range(10)], 
                    'solver': ['lbfgs', 'sgd', 'adam'], 
#                        'momentum': 0.9, 
                    'learning_rate': ['constant', 'invscaling', 'adaptive'], 
                    'early_stopping': [True, False],
                }),
#        'TRE': (tree.DecisionTreeClassifier(),{}),
#        'GBC': (GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0),{}),
#        'RFC': (RandomForestClassifier(),{}),
    }


trained_clfs = []

for name, clf_data in classifiers.iteritems():
    clf = clf_data[0]
    params = clf_data[1]
    rand_scv = RandomizedSearchCV(clf, params, n_iter=20)
    if params:
        fitted_clf = rand_scv.fit(training_data, training_target)
    else:
        fitted_clf = clf.fit(training_data, training_target)

    trained_clfs.append((name, fitted_clf))


#scores = [(n, clf.fit(training_data, training_target).score(test_data,
#    test_target)) for n, clf in classifiers.iteritems()]

for name, clf in sorted(trained_clfs, key=lambda t: t[1], reverse=True):
    print name, clf.score(test_data, test_target)

#for n, clf in trained_clfs:
#    print "CLASSIFIER: " + n
#    gesture_pred = clf.predict(test_data)
#    target_names = list(string.ascii_lowercase)
#    cm = confusion_matrix(test_target, gesture_pred, target_names)
#    utils.plot_confusion_matrix(cm, classes=target_names, title=n)
#
#
#    print(classification_report(test_target, gesture_pred, target_names=target_names))

import winsound
winsound.Beep(500,500)
winsound.Beep(500,500)
