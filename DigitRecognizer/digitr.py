#!/usr/bin/python
from scipy import *
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from sklearn import svm

def load ():
      # Load a csv of floats:
      #train = np.genfromtxt("data/train.csv", delimiter=",", skip_header=1)
      #y_train = train[:,0].astype(int)
      #x_train = train[:,1:]

      npzfile = np.load('data/bindata.npz')
      x = npzfile['x']
      y = npzfile['y'].astype(int)

      #test = np.genfromtxt("data/test.csv", delimiter=",", skip_header=1)
      #x_test = test
      
      return y, x

def main ():
    y, x, = load();

    n_samples = x.shape[0]
    n_features = x.shape[1]
    n_classes = 10

    print("Total dataset size:")
    print("n_samples: %d" % n_samples)
    print("n_features: %d" % n_features)
    print("n_classes: %d" % n_classes)

    # split into a training and testing set
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

    x_train = x[0:1000,:]
    y_train = y[0:1000]
    x_test = x[1000:2000,:]
    y_test = y[1000:2000]

    # Set the parameters by cross-validation
    C=[]
    gamma=[]
    for i in range(21): C.append(10.0**(i-5))
    for i in range(17): gamma.append(10**(i-14))

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': gamma, 'C': C},
                        {'kernel': ['linear'], 'C': C}]

    clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=5, scoring='accuracy')

    # We learn the digits on the first half of the digits
    clf.fit(x_train, y_train)

    # Now predict the value of the digit on the second half:
    y_true, y_pred = y_test, clf.predict(x_test)
    
    print"Best parameters set found on development set:"
    print
    print clf.best_estimator_
    print
    print "Grid scores on development set:"
    print
    for params, mean_score, scores in clf.grid_scores_:
        print ("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params))
    print

    print("Classification report for classifier %s:\n%s\n"
          % (clf.best_estimator_, metrics.classification_report(y_true, y_pred)))

    print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
