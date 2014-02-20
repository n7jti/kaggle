#!/usr/bin/python
from scipy import *
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from sklearn.svm import SVC
from sklearn import datasets

def load ():
      # Load a csv of floats:
      train = np.genfromtxt("data/train.csv", delimiter=",", skip_header=1)
      y_train = train[:,0]
      x_train = train[:,1:]

      test = np.genfromtxt("data/test.csv", delimiter=",", skip_header=1)
      x_test = test
      
      return y_train,x_train, x_test
def main ():
    print "loading!"
    y, x, x_test = load();
    print "saving"
    #np.savez('data/bindata', x=x, y=y)
    #print "done"

#   npzfile = np.load('data/bindata.npz')
#   x = npzfile['x']
#   y = npzfile['y']
#
#   print 'x', x.shape, 'y', y.shape

   
if __name__ == "__main__":
    main()
