from sklearn import svm, neighbors, tree, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, VarianceThreshold
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


if __name__=="__main__":
    num_features = 200
#    train_paths = [os.path.join("Leap_Data", "DataGath1"), os.path.join("Leap_Data", "DataGath3"), os.path.join("Leap_Data", "Participant 0")]
    test_paths = [os.path.join("Leap_Data", "DataGath2")]
    train_paths = []
    auto_split=True
    training_data_all, test_data_all, training_target, test_target = get_train_test_split(train_paths=train_paths, test_paths=test_paths, use_auto_split=auto_split, frames_per_gesture=2, separate_frames=False, feature_set_type="all")
    
    
    c_range = [2**(x) for x in range(-5, 15)] # http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf page 5
    gamma_range = [2**(x) for x in range(-15, 3)] # http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf page 5
    desc_functions = ['ovo', 'ovr']
    
    # maybe drop poly and sigmoid
    svm_params = [
                    {'kernel': ['linear'], 
                    'C': c_range, 
                    'decision_function_shape': desc_functions},
    
                    {'kernel': ['rbf'], 
                    'gamma': gamma_range, 
                    'C': c_range, 
                    'decision_function_shape': desc_functions},
    
                    {'kernel': ['poly'], 
                    'gamma': gamma_range, 
                    'degree': range(5), # https://stackoverflow.com/questions/26337403/what-is-a-good-range-of-values-for-the-svm-svc-hyperparameters-to-be-explored
                    # not tuning coef0 https://stackoverflow.com/questions/21390570/scikit-learn-svc-coef0-parameter-range                
                    'C': c_range, 
                    'decision_function_shape': desc_functions},
    
                    # sigmoid is easily invalid, drop sigmoid
                ]
                
    neighbor_range = range(1,50) # maybe higher with more data?
    weight_opts = ['uniform', 'distance']
    p_range = range(1,10) # special for one and 2, minkwski shit?? arbitrary p for sparse matrices 
                            #http://scikit-learn.org/stable/modules/neighbors.html#classification
    # https://datascience.stackexchange.com/questions/989/svm-using-scikit-learn-runs-endlessly-and-never-completes-execution
    knn_params = [
    # brute algortihm infeasible for large data
   
                    {'n_neighbors': neighbor_range, 
                    'weights': weight_opts,
                    'algorithm': ['auto', 'ball_tree'],
                    'p': p_range,}
                    # kd tree bad for high dimensionality, consider with lower dimensions
                    # leaf_size is more or less meaningless for accuracy, performance paramter
                ]
                
                
    nodes_per_layer_range = range(26, num_features, 5)
    alpha_range = [10**x for x in range(-8,3)] # regularization term: margin width kinda thing, prevent overfitting
    learning_rate_init_range = [10**x for x in range(-6,0)] # http://www.uio.no/studier/emner/matnat/ifi/INF3490/h15/beskjeder/question-about-mlp-learning-rate.html 
    
    mlp_params = {
                    'hidden_layer_sizes': [(x,) for x in nodes_per_layer_range],
                    'activation': ('identity', 'logistic', 'tanh', 'relu'), 
                    'learning_rate_init': learning_rate_init_range, 
                    'alpha': alpha_range, 
                    'solver': ['lbfgs', 'sgd', 'adam'], 
                    'learning_rate': ['constant', 'invscaling', 'adaptive'], 
                }
                
#    mlp_params = {}
    
    log.info("svm_params: {}, \n knn_params: {}, \n mlp_params: {}".format(svm_params, knn_params, mlp_params))
    
    classifiers = {        
#            'SVM': (svm.SVC(), svm_params),
    #        'BNB': (BernoulliNB(), {}), # alpha not important because we have well defined priors https://stats.stackexchange.com/questions/108797/in-naive-bayes-why-bother-with-laplacian-smoothing-when-we-have-unknown-words-i
    #        'GNB': (GaussianNB(), {}), # binarize parameter may work for finger extension
    #       MultinomialNB is for positives only
#            'kNN': (neighbors.KNeighborsClassifier(), knn_params),
    #        'rNN': (neighbors.RadiusNeighborsClassifier(), rnn_params), # not effective in higher dimensions
            'MLP': (MLPClassifier(), mlp_params),
    #        'TRE': (tree.DecisionTreeClassifier(),{}),
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
    
    selector = SelectKBest(k=num_features)
    training_data = selector.fit_transform(training_data, training_target)
    test_data = selector.transform(test_data)
    
    
    log.info(selector)
    log.info("number of features: {}".format(len(training_data[0])))
    print("number of features: {}".format(len(training_data[0])))
    
    
    trained_clfs = []
    
    for name, clf_data in classifiers.iteritems():
        clf = clf_data[0]
        params = clf_data[1]
#        rand_scv = RandomizedSearchCV(clf, params, n_iter=20, n_jobs=-1)
        rand_scv = GridSearchCV(clf, params)
        if params:
            fitted_clf = rand_scv.fit(training_data, training_target)
            log.info("{} chosen features: {}".format(name, fitted_clf.best_params_))
    
        else:
            fitted_clf = clf.fit(training_data, training_target)
            log.info("{} default features: {}".format(name, fitted_clf.get_params()))
    
        trained_clfs.append((name, fitted_clf))
    
    
    
    for n, clf in trained_clfs:
        gesture_pred = clf.predict(test_data)
        target_names = list(string.ascii_lowercase)
        score = clf.score(test_data, test_target)
        print "CLASSIFIER: {} {}".format(n, score)
        log.info("CLASSIFIER: {} {}".format(n, score))
    
        report = classification_report(test_target, gesture_pred, target_names=target_names)
        cm = confusion_matrix(test_target, gesture_pred, target_names)
    #    utils.plot_confusion_matrix(cm, classes=target_names, title=n)
        log.info(report)
        log.info(cm)
    
    
    import winsound
    winsound.Beep(500,500)
    winsound.Beep(500,500)
