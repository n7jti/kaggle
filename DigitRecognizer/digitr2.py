#!/usr/bin/python
from scipy import *
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from sklearn import svm
import time 
import pickle

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
    print 'starting', time.asctime(time.localtime())
    start = time.clock()
    y, x, = load();

    # split into a training and testing set
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

    x_train = x[0:20000,:]
    y_train = y[0:20000]
    x_test = x[20000:40000,:]
    y_test = y[20000:40000]

    # Set the parameters by cross-validation
    C=10
    gamma=1e-7
    
    clf = svm.SVC(C=C, gamma=gamma)

    # We learn the digits on the first half of the digits
    clf.fit(x_train, y_train)

    # Pickle the model!
    outf = open('model.pkl', 'wb')
    pickle.dump(clf, outf)
    outf.close()

    # Now predict the value of the digit on the second half:
    y_true, y_pred = y_test, clf.predict(x_test)

    print("Classification report for classifier %s:\n%s\n"
          % (clf, metrics.classification_report(y_true, y_pred)))

    print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_true, y_pred))

    stop = time.clock()

    print 'elapsed:', stop - start

if __name__ == "__main__":
    main()
