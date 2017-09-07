from sklearn import svm, neighbors, tree, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.feature_selection import SelectKBest, VarianceThreshold, RFE, SelectFromModel, mutual_info_classif
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier, RandomForestClassifier
import os, string, time, logging, pickle
from read_data import read_data, get_feature_names, load_selected_features
import numpy as np


log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)
filename = "experiment_" + str(time.time()) + ".txt"
path_to_logs = os.path.abspath("logs/")
path_to_log_file = os.path.join(path_to_logs, filename)
fh = logging.FileHandler(path_to_log_file)
fh.setLevel(logging.DEBUG)
log.addHandler(fh)


def select_features(training_data, training_target, test_data, feature_labels, fresh=False):
    # remove featuers with 0 variance, only changes at threshold 1.... check with more data  
    training_data, test_data, feature_labels = remove_0_var(training_data, training_target, test_data, feature_labels)
    training_data, test_data = scale(training_data, test_data)
    
    if fresh:
        selector = SelectKBest(k=500)
        training_data = selector.fit_transform(training_data, training_target)
        test_data = selector.transform(test_data)
        feature_labels = selector.transform(feature_labels)
        log.info(selector)

        selector = SelectFromModel(ExtraTreesClassifier(), threshold=0.002)
        training_data = selector.fit_transform(training_data, training_target)
        test_data = selector.transform(test_data)
        feature_labels = selector.transform(feature_labels)
        log.info(selector)

    else:
    
        loaded_labels = list(load_selected_features())
        feature_indeces = [x for x, feature in enumerate(feature_labels[0]) if feature in loaded_labels]
        feature_labels = [[each_list[i] for i in feature_indeces] for each_list in feature_labels]
        training_data = [[each_list[i] for i in feature_indeces] for each_list in training_data]
        test_data = [[each_list[i] for i in feature_indeces] for each_list in test_data]

        log.info("last run features")


    log.info("number of features: {}".format(len(training_data[0])))
    print("number of features: {}".format(len(training_data[0])))
    log.info("features selected: {}".format(feature_labels[0]))
#    print("features selected: {}".format(feature_labels[0]))    

    
    return training_data, test_data, feature_labels

    

def optimize_params(clf, params, training_data, training_target):
    try:
        rand_scv = RandomizedSearchCV(clf, params, n_iter=40, n_jobs=-1)
        return rand_scv.fit(training_data, training_target)
    except ValueError:
        rand_scv = GridSearchCV(clf, params)
        return rand_scv.fit(training_data, training_target)
        

def test_clf(name, clf, test_data, test_target, results):
    gesture_pred = clf.predict(test_data)
    target_names = list(string.ascii_lowercase)
    accuracy = accuracy_score(test_target, gesture_pred)
    precision = precision_score(test_target, gesture_pred, average='weighted')
    recall = recall_score(test_target, gesture_pred, average='weighted')
    results[name] = {}
    results[name]['accuracy'] = accuracy
    results[name]['precision'] = precision
    results[name]['recall'] = recall
    
    print "CLASSIFIER: {} {}".format(name, accuracy)
    log.info("CLASSIFIER: {} {}".format(name, accuracy))

    report = classification_report(test_target, gesture_pred, target_names=target_names)
    cm = confusion_matrix(test_target, gesture_pred, target_names)
#    utils.plot_confusion_matrix(cm, classes=target_names, title=n)
    log.info(report)
    log.info(cm)

def remove_0_var(training_data, training_target, test_data, feature_labels):
    selector = VarianceThreshold()
    training_data = selector.fit_transform(training_data, training_target)
    test_data = selector.transform(test_data)
    feature_labels = selector.transform(feature_labels)
    return training_data, test_data, feature_labels

def scale(training_data, test_data):
    log.info("scaling")
    training_data = preprocessing.scale(training_data)
    test_data = preprocessing.scale(test_data)
    return training_data, test_data

def load_paths(paths, fresh, frames_per_gesture, separate_frames, feature_set_type):
    all_data = []
    all_target = []
    for path in paths:
#        print("loading path {}".format(path))
        if fresh:
            data, target = read_data(path, frames_per_gesture, separate_frames, feature_set_type)
            try:
                with open(path[:-4] + "Participant.data", 'wb') as fp:
                    pickle.dump((data, target), fp)
            except IOError:
                continue
        else:
            try:
                with open(path[:-4] + "Participant.data", 'rb') as fp:
                    data, target = pickle.load(fp)
            except IOError:
                continue
        all_data.extend(data)
        all_target.extend(target)
    return all_data, all_target

def get_train_test_split(frames_per_gesture, separate_frames, fresh=False, feature_set_type="all", train_paths=[], test_paths=[], use_auto_split=False, average=False):

    log.info('Data variables: \n'
            '\t train_paths: {}, \n'
            '\t test_paths: {}, \n'
            '\t use_auto_split: {}, \n'
            '\t frames_per_gesture: {}, \n'
            '\t separate_frames: {}, \n'
            '\t feature_set_type: {} \n'
            '\t average: {}'
            .format(train_paths, 
                    test_paths, 
                    use_auto_split, 
                    frames_per_gesture, 
                    separate_frames, 
                    feature_set_type,
                    average))

    training_data, training_target = load_paths(train_paths, fresh, frames_per_gesture, separate_frames, feature_set_type)
    test_data, test_target = load_paths(test_paths, fresh, frames_per_gesture, separate_frames, feature_set_type)

    if use_auto_split:
        data = test_data + training_data
        target = test_target + training_target
        training_data, test_data, training_target, test_target = train_test_split(data, target, test_size=0.25, random_state=0)

    return training_data, test_data, training_target, test_target


    
def run_experiment(test_participant, fpg=2, quick_test=False):
    results = {}    
    
    train_paths = [os.path.join("Leap_Data", "Legit_Data", "Participant " + str(x), "Leap") for x in range(0, test_participant) + range(test_participant+1,50)]
    test_paths = [os.path.join("Leap_Data", "Legit_Data", "Participant " + str(test_participant), "Leap")]
    training_data, test_data, training_target, test_target = get_train_test_split(
            train_paths= [] if quick_test else train_paths, 
            test_paths=test_paths, 
            use_auto_split=not quick_test, 
            frames_per_gesture=fpg, 
            separate_frames=False, 
            feature_set_type='all',
            average=False,
            fresh=True,
        )
    
    all_feature_labels = get_feature_names(test_paths[0], 'all') * fpg
    start = time.clock()
    training_data, test_data, feature_labels = select_features(training_data, training_target, test_data, [all_feature_labels], fresh=True)
    end = time.clock()
    log.info("feature selection took {} seconds".format(end - start))
    print("feature selection took {} seconds".format(end - start))



    c_range = [2**(x) for x in range(-5, 15)] # http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf page 5
    lower_c = [2**(x) for x in range(-5, 7)]
    gamma_range = [2**(x) for x in range(-15, 3)] + ['auto'] # http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf page 5
    
    svm_params = {  'gamma': gamma_range, 
                    'C': lower_c,}
    
    # maybe drop poly and sigmoid
#    svm_params = [
#                    {'kernel': ['linear'], 
#                    'C': c_range, 
#                    'decision_function_shape': desc_functions},
#    
#
#    
#                    {'kernel': ['poly'], 
#                    'gamma': gamma_range, 
#                    'degree': range(5), # https://stackoverflow.com/questions/26337403/what-is-a-good-range-of-values-for-the-svm-svc-hyperparameters-to-be-explored
#                    # not tuning coef0 https://stackoverflow.com/questions/21390570/scikit-learn-svc-coef0-parameter-range                
#                    'C': c_range, 
#                    'decision_function_shape': desc_functions},
#                    # poly was actually doing better, gridsearch lied??
#                    # tiny C 0.001 stuff
##                    
#                    {'kernel': ['rbf'], 
#                    'gamma': gamma_range, 
#                    'C': c_range, 
#                    'decision_function_shape': desc_functions},
#                    # small C 6 stuff                    
##                    
##                     sigmoid is easily invalid, drop sigmoid
#                ]
    
                
    neighbor_range = range(1,50) # maybe higher with more data?
    p_range = range(1,5) # special for one and 2, minkwski shit?? arbitrary p for sparse matrices 
                            #http://scikit-learn.org/stable/modules/neighbors.html#classification
    # https://datascience.stackexchange.com/questions/989/svm-using-scikit-learn-runs-endlessly-and-never-completes-execution
    # brute algortihm infeasible for large data
    # kd tree bad for high dimensionality, consider with lower dimensions
    # leaf_size is more or less meaningless for accuracy, performance paramter
    knn_params = {'n_neighbors': neighbor_range, 
                    'algorithm': ['auto', 'ball_tree'],
                    'p': p_range,}
                    
                
    
    if 26 > len(training_data[0]):
        nodes_per_layer_range = range(len(training_data[0]), 26, 5)
    else:
        nodes_per_layer_range = range(26, len(training_data[0]), 5)
        
#    nodes_per_layer_range = range(1, 50)
    alpha_range = np.logspace(-5, 3, 5) # regularization term: margin width kinda thing, prevent overfitting
    learning_rate_init_range = [10**x for x in range(-6,0)] # http://www.uio.no/studier/emner/matnat/ifi/INF3490/h15/beskjeder/question-about-mlp-learning-rate.html 
    #radius neighbours not effective in higher dimensions
    mlp_params = {
                    'hidden_layer_sizes': [(x,) for x in nodes_per_layer_range],
                    'activation': ('identity', 'logistic', 'tanh', 'relu'),
                    'activation': ('logistic', 'tanh'),
                    'learning_rate_init': learning_rate_init_range,
                    'alpha': alpha_range,
                    'solver': ['sgd', 'adam'],
                    'learning_rate': ['constant', 'invscaling', 'adaptive'],
                }
                
#            'BNB': (BernoulliNB(), {}), # alpha not important because we have well defined priors https://stats.stackexchange.com/questions/108797/in-naive-bayes-why-bother-with-laplacian-smoothing-when-we-have-unknown-words-i
#            'GNB': (GaussianNB(), {}), # binarize parameter may work for finger extension, MultinomialNB is for positives only
                
                
#    log.info("svm_params: {}, \n knn_params: {}, \n mlp_params: {}".format(svm_params, knn_params, mlp_params))
    
    classifiers = {        
#            'SVM': (svm.SVC(probability=True, kernel='rbf', decision_function_shape='ovo'), svm_params),
#            'SVM no tuning': (svm.SVC(probability=True, kernel='rbf', decision_function_shape='ovo'), {}),

#            'kNN': (neighbors.KNeighborsClassifier(weights='distance'), knn_params),
#            'kNN no tuning': (neighbors.KNeighborsClassifier(weights='distance'), {}),

            'MLP': (MLPClassifier(), mlp_params),
#            'MLP no tuning': (MLPClassifier(), {}),

#            'ETC': (ExtraTreesClassifier(), {}),
#            'RFC': (RandomForestClassifier(), {}),
        }
    
    
    trained_clfs = []
    
    for name, clf_data in classifiers.iteritems():
        clf = clf_data[0]
        params = clf_data[1]
              
        start = time.clock()
        fitted_clf = optimize_params(clf, params, training_data, training_target)
        end = time.clock()
        log.info("parameter tuning {} took {} seconds".format(name, end - start))
        print("parameter tuning for {} took {} seconds".format(name, end - start))

        log.info("{} chosen parameters: {}".format(name, fitted_clf.best_params_))
        print("{} chosen parameters: {}".format(name, fitted_clf.best_params_))
    
        trained_clfs.append((name, fitted_clf.best_estimator_))
        
        start = time.clock()
        test_clf(name, fitted_clf, test_data, test_target, results)
        end = time.clock()
        log.info("testing classifier {} took {} seconds".format(name, end - start))
        print("testing classifier {} took {} seconds".format(name, end - start))
    
    
    voting_clf = VotingClassifier(estimators=trained_clfs, voting='soft')
    voting_clf.fit(training_data, training_target)

    
    start = time.clock()
    test_clf("voting", voting_clf, test_data, test_target, results)
    end = time.clock()
    log.info("testing classifier {} took {} seconds".format("voting", end - start))
    print("testing classifier {} took {} seconds".format("voting", end - start))

    return results

if __name__=="__main__":
#    valid_participants = range(3) + [x for x in range(12, 49) if x not in [13, 20, 24, 25, 34]]
    valid_participants = [12]
    all_results = []
    for x in valid_participants:
        all_results.append(run_experiment(x))
        
    print(all_results)
        
    import winsound
    winsound.Beep(500,500)
    winsound.Beep(500,500)
    