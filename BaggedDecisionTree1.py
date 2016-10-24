# Bagged Decision Trees for Classification
import pandas
import time
import numpy as np

from sklearn import cross_validation
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

#load data from external source
start_time = time.time()
dataframe = pandas.read_csv(url, names=names)
print("--- time to load data from url %s seconds ---" % (time.time() - start_time))
print "dataframe.size=", dataframe.size, "dataframe.shape=", dataframe.shape
#dataframe.size 6912 dataframe.shape (768, 9)

array = dataframe.values
print "type(array)", type(array)
print "array.shape=", array.shape
#<type 'numpy.ndarray'>, (768L, 9L)

#subset dataframe columns we want to look at.
X = array[:,0:8]
Y = array[:,8]

print "type(X)=", type(X), "X.shape=", X.shape #<type 'numpy.ndarray'> X.shape= (768L, 8L)
print "type(Y)=", type(Y), "Y.shape=", Y.shape #<type 'numpy.ndarray'> Y.shape= (768L,)

#set parameters for model construction
#num_folds = 100
num_folds = 200
num_instances = len(X)
seed = 7

start_time = time.time()
kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
print("--- time to cross_validation.KFold %s seconds ---" % (time.time() - start_time))

start_time = time.time()
cart = DecisionTreeClassifier()
print("--- time to create DecisionTreeClassifier %s seconds ---" % (time.time() - start_time))
num_trees = 100

start_time = time.time()
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
print("--- time to create BaggingClassifier %s seconds ---" % (time.time() - start_time))

start_time = time.time()
results = cross_validation.cross_val_score(model, X, Y, cv=kfold)
print("--- time to execute  cross_validation.cross_val_score %s seconds ---" % (time.time() - start_time))
print "type(results)=", type(results), "results.size=", results.size, "results.shape=", results.shape
#<type 'numpy.ndarray'>, results.size= num_folds as used in cross_validation.KFold

print "cross_validation.cross_val_score.mean () = results.mean()=", results.mean()
print "cross_validation.cross_val_score.max () = results.max()=", results.max()
print "cross_validation.cross_val_score.min () = results.min()=", results.min()
print "cross_validation.cross_val_score variances = ", np.var(results)
print "cross_validation.cross_val_score standard variation = ", np.std(results)

temp = list(results)
print type(temp), len(temp)

import matplotlib.pyplot as plt
x = range(len(temp))
y = list(results)

plt.scatter(x, y, alpha=0.5)
plt.show()

plt.xlabel('mean(cross_val_score)')
plt.ylabel('Frequency')
plt.title('Bagged Decision Tree for pima-indians-diabetes.data.')
plt.hist(y)
plt.show()

#NB: as num_folds is increased we get more plot points on the charts to show the range and distribution of results.
# as expected, when a large enough size is chosen the histogram starts to resemble a bell curve.
# oddly the bell curve shows a peak around 0.8 but zero results @ this maximum.

"""
based code from here, added stuff for diagnosis and training purposes.
http://machinelearningmastery.com/ensemble-machine-learning-algorithms-python-scikit-learn/


"""