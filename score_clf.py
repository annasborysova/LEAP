from sklearn import svm, neighbors, tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
import process, string, utils
from sklearn.metrics import classification_report, confusion_matrix


training_data, test_data, training_target, test_target = train_test_split(process.gesture_data, process.gesture_names, test_size=0.25, random_state=0)

classifiers = {
        'SVCP': svm.SVC(gamma=0.001, C=10),
        'SVCR': svm.SVC(gamma=0.0001, C=50),
        'NB ': GaussianNB(),
        'BNB': BernoulliNB(),
        'NBU': neighbors.KNeighborsClassifier(5, weights='uniform'),
        'NBD': neighbors.KNeighborsClassifier(5, weights='distance'),
        'TRE': tree.DecisionTreeClassifier(),
        'GBC': GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0),
        'RFC': RandomForestClassifier(),
        'MLP': MLPClassifier(),
    }

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
    
