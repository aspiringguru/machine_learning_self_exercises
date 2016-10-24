# Random Forest Classification
import pandas
import time
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
num_trees = 100
max_features = 3
#-------------- configure model end ----------------
start_time = time.time()
kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
print("--- time to execute  cross_validation.KFold %s seconds ---" % (time.time() - start_time))

start_time = time.time()
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
print("--- time to execute  RandomForestClassifier %s seconds ---" % (time.time() - start_time))

start_time = time.time()
results = cross_validation.cross_val_score(model, X, Y, cv=kfold)
print("--- time to execute  cross_validation.cross_val_score %s seconds ---" % (time.time() - start_time))
print "type(results)=", type(results)
print "results.mean()=", results.mean()
print "results.max()=", results.max()
print "results.min()=", results.min()

import matplotlib.pyplot as plt
x = range(len(list(results)))
y = list(results)

plt.scatter(x, y, alpha=0.5)
plt.show()

plt.xlabel('mean(cross_val_score)')
plt.ylabel('Frequency')
plt.title('Rando Forest for pima-indians-diabetes.data.')
plt.hist(y)
plt.show()


