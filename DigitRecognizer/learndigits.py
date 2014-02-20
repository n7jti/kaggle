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
    y, x, = load();

    # split into a training and testing set
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

    x_train = x
    y_train = y

    # Set the parameters by cross-validation
    C=10
    gamma=1e-7
    
    clf = svm.SVC(C=C, gamma=gamma)

    # We learn the digits on the first half of the digits
    clf.fit(x_train, y_train)

    # Pickle the model!
    outf = open('training.pkl', 'wb')
    pickle.dump(clf, outf)
    outf.close()

    print 'done!', time.asctime(time.localtime())

if __name__ == "__main__":
    main()
