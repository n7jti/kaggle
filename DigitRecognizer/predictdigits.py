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
    test = np.genfromtxt("data/test.csv", delimiter=",", skip_header=1)

    return test
      

def main ():
    print 'starting', time.asctime(time.localtime())
    x = load();
    print 'done loading, unpickling', time.asctime(time.localtime())

    # un-Pickle the model!
    inf = open('training.pkl', 'rb')
    clf = pickle.load(inf)
    inf.close()

    print 'predicting', time.asctime(time.localtime())
    y_hat = clf.predict(x)

    print 'writing out answer', time.asctime(time.localtime())
    f=open('submitme.cv', 'w')
    f.write('ImageId, Label\n')

    for idx in range(y_hat.shape[0]):
        f.write(str(idx+1))
        f.write(',')
        f.write(str(y_hat[idx]))
        f.write('\n')


    print 'done!', time.asctime(time.localtime())

if __name__ == "__main__":
    main()
