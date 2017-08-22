from sklearn import svm, neighbors, tree, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, VarianceThreshold, RFE, chi2
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import os, string, utils, time, logging
from read_data import read_data

log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)
filename = "experiment_" + str(time.time()) + ".txt"
path_to_logs = os.path.abspath("logs/")
path_to_log_file = os.path.join(path_to_logs, filename)
fh = logging.FileHandler(path_to_log_file)
fh.setLevel(logging.DEBUG)
log.addHandler(fh)



def get_train_test_split(frames_per_gesture, separate_frames, feature_set_type="all", train_paths=[], test_paths=[], use_auto_split=False):
    
    log.info('Data variables: \n'
            '\t train_paths: {}, \n'
            '\t test_paths: {}, \n'
            '\t use_auto_split: {}, \n'
            '\t frames_per_gesture: {}, \n'
            '\t separate_frames: {}, \n'
            '\t feature_set_type: {}'
            .format(train_paths, 
                    test_paths, 
                    use_auto_split, 
                    frames_per_gesture, 
                    separate_frames, 
                    feature_set_type))

    training_data = []
    training_target = []
    for path in train_paths:
        data, target = read_data(path, frames_per_gesture, separate_frames, feature_set_type)
        training_data.extend(data)
        training_target.extend(target)
        
    test_data = []
    test_target = []
    for path in test_paths:
        data, target = read_data(path, frames_per_gesture, separate_frames, feature_set_type)
        test_data.extend(data)
        test_target.extend(target)

    if use_auto_split:
        data = test_data + training_data
        target = test_target + training_target
        training_data, test_data, training_target, test_target = train_test_split(data, target, test_size=0.25, random_state=0)

    return training_data, test_data, training_target, test_target


train_paths = [os.path.join("Leap_Data", "DataGath1"), os.path.join("Leap_Data", "DataGath3"), os.path.join("Leap_Data", "Participant 0")]
test_paths = [os.path.join("Leap_Data", "DataGath2")]
#test_paths = []
training_data_all, test_data_all, training_target, test_target = get_train_test_split(train_paths=train_paths, test_paths=test_paths, use_auto_split=False, frames_per_gesture=1, separate_frames=True, feature_set_type="all")


svm_params = {
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
            }
            
nb_params = {
                'alpha': [1.0/(10**x) for x in range(50)],
                'fit_prior': [True, False],                    
            }
            
knn_params = {
                'n_neighbors': range(1,10),
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'leaf_size': range(1, 100, 10),
                'p': range(1,10),
            }
            
mlp_params = {
#               'beta_1': [9.0/(10**x) for x in range(50)], 
#                'warm_start': [True, False],
#               'beta_2': [9.0/(10**x) for x in range(50)],
#                'shuffle': [True, False],
#                'verbose': [True, False],
#                'nesterovs_momentum': [True, False], 
#               'hidden_layer_sizes': [(100,),], # add more values
#               'epsilon': 1e-08, 
                'activation': ('identity', 'logistic', 'tanh', 'relu'), 
#               'batch_size': 'auto', 
#               'power_t': 0.5, 
#               'random_state': None, 
                'learning_rate_init': [1.0/(10**x) for x in range(10)], 
                'tol': [1.0/(10**x) for x in range(50)], # likes to be 50/high
                'validation_fraction': [1.0/(10**x) for x in range(1,10)], 
                'alpha': [1.0/(10**x) for x in range(10)], 
                'solver': ['lbfgs', 'sgd', 'adam'], 
#               'momentum': 0.9, 
                'learning_rate': ['constant', 'invscaling', 'adaptive'], 
#                'early_stopping': [True, False],
            }
            
mlp_params = {}

log.info("svm_params: {}, \n nb_params: {}, \n knn_params: {}, \n mlp_params: {}".format(svm_params, nb_params, knn_params, mlp_params))

classifiers = {        
        'SVM': (svm.SVC(), svm_params),
        'BNB': (BernoulliNB(), nb_params),
        'kNN': (neighbors.KNeighborsClassifier(), knn_params),
        'MLP': (MLPClassifier(), mlp_params),
#        'TRE': (tree.DecisionTreeClassifier(),{}),
#        'GBC': (GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0),{}),
#        'RFC': (RandomForestClassifier(),{}),
    }

# remove featuers with 0 variance, only changes at threshold 1.... check with more data

selector = VarianceThreshold()
training_data = selector.fit_transform(training_data_all, training_target)
test_data = selector.transform(test_data_all)


# normalize
normalize = True
log.info("normalize: {}".format(normalize))
if normalize:
    training_data = preprocessing.scale(training_data)
    test_data = preprocessing.scale(test_data)

selector = SelectKBest(k=200)
#selector = PCA(n_components=3)


#training_data = selector.fit_transform(training_data, training_target)
#test_data = selector.transform(test_data)
#
log.info(selector)
log.info("number of features: {}".format(len(training_data[0])))
print("number of features: {}".format(len(training_data[0])))


trained_clfs = []

for name, clf_data in classifiers.iteritems():
    clf = clf_data[0]
    params = clf_data[1]
    # feature selection
#    rfe = RFE(model, 1114)
    rand_scv = RandomizedSearchCV(clf, params, n_iter=20)
    if params:
        fitted_clf = rand_scv.fit(training_data, training_target)
    else:
        fitted_clf = clf.fit(training_data, training_target)

    trained_clfs.append((name, fitted_clf))
    log.info(name + " " + str(fitted_clf.get_params()))



for n, clf in trained_clfs:
    gesture_pred = clf.predict(test_data)
    target_names = list(string.ascii_lowercase)
    score = clf.score(test_data, test_target)
    print "CLASSIFIER: {} {}".format(n, score)
    log.info("CLASSIFIER: {} {}".format(n, score))

    report = classification_report(test_target, gesture_pred, target_names=target_names)
#    cm = confusion_matrix(test_target, gesture_pred, target_names)
#    utils.plot_confusion_matrix(cm, classes=target_names, title=n)
    log.info(report)


import winsound
winsound.Beep(500,500)
winsound.Beep(500,500)
