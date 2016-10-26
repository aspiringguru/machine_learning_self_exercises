"""
http://scikit-learn.org/stable/modules/cross_validation.html
working example
"""

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()
print "iris.data.shape=", iris.data.shape, "type(iris.data)=", type(iris.data)
print "iris.target.shape", iris.target.shape, "type(iris.target)=", type(iris.target)

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

print "type(X_train)=", type(X_train), "X_train.shape=", X_train.shape
print "type(X_test)=", type(X_test), "X_test.shape=", X_test.shape

