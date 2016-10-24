# Random Forest Classification
import pandas
import time
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]

#-------------- configure model start ----------------
num_folds = 200
num_instances = len(X)
seed = 7
num_trees = 200
max_features = 3
#-------------- configure model end ----------------
start_time = time.time()
kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
print("--- time to execute  cross_validation.KFold %s seconds ---" % (time.time() - start_time))
#http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
#K-Folds cross-validator
# Provides train/test indices to split data in train/test sets.
# Split dataset into k consecutive folds (without shuffling by default).
# Each fold is then used once as a validation while the k - 1 remaining folds form the training set.

start_time = time.time()
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
print("--- time to execute  RandomForestClassifier %s seconds ---" % (time.time() - start_time))
#http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# A random forest is a _meta estimator_ that fits a number of decision tree classifiers on various sub-samples
# of the dataset and use averaging to improve the predictive accuracy and control over-fitting.
# The sub-sample size is always the same as the original input sample size but the samples are drawn
# with replacement if bootstrap=True (default).



start_time = time.time()
results = cross_validation.cross_val_score(model, X, Y, cv=kfold)
print("--- time to execute  cross_validation.cross_val_score %s seconds ---" % (time.time() - start_time))
#http://scikit-learn.org/stable/modules/cross_validation.html
print "type(results)=", type(results)
print "results.mean()=", results.mean()
print "results.max()=", results.max()
print "results.min()=", results.min()
print "cross_validation.cross_val_score variances = ", np.var(results)
print "cross_validation.cross_val_score standard deviation = ", np.std(results)
print "mean +/- 1 std dev = ", results.mean() - np.std(results), ":", results.mean() + np.std(results)

import matplotlib.pyplot as plt
x = range(len(list(results)))
y = list(results)

plt.scatter(x, y, alpha=0.5)
plt.show()

plt.xlabel('mean(cross_val_score)')
plt.ylabel('No of results')
plt.title('Random Forest for pima-indians-diabetes.data.')
plt.hist(y)
plt.show()


