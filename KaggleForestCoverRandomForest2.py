"""
https://www.kaggle.com/triskelion/forest-cover-type-prediction/first-try-with-random-forests/code
https://archive.ics.uci.edu/ml/datasets/Covertype

Todo : predict the forest cover type (the predominant kind of tree cover) from strictly cartographic variables
(as opposed to remotely sensed data).



"""

import time
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import ensemble
from sklearn.metrics import accuracy_score
import numpy as np

raw_data = "./covtype/covtype.data"

df = pd.read_csv(raw_data, header=None)
print "type(df)=", type(df)
print "df.shape=", df.shape
#print "df.head(5)=\n", df.head(5)

#manually creating col_names since these are not in the .cvs file based on covtype.info
col_names = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', \
             'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', \
             'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', \
             'Wilderness_Area4', 'ST1', 'ST2', 'ST3', 'ST4', 'ST5', 'ST6', 'ST7', 'ST8', 'ST9', 'ST10', 'ST11', \
             'ST12', 'ST13', 'ST14', 'ST15', 'ST16', 'ST17', 'ST18', 'ST19', 'ST20', 'ST21', 'ST22', 'ST23', 'ST24', \
             'ST25', 'ST26', 'ST27', 'ST28', 'ST29', 'ST30', 'ST31', 'ST32', 'ST33', 'ST34', 'ST35', 'ST36', 'ST37', \
             'ST38', 'ST39', 'ST40', 'Cover_Type']

print type(col_names), len(col_names)
print "col_names", col_names
#NB len(col_names)=55 & df.shape=(581012, 55), so # of columns matches.
df.columns = col_names
print "df.head(5)=\n", df.head(5)

#now split dataframe into columns for features & keys
keys = df['Cover_Type']
print "type(keys)", type(keys)
print "keys.shape=", keys.shape
df = df.drop('Cover_Type', 1)
print "df.shape=", df.shape
print "list(df.columns)", list(df.columns)
#print "df.head(5)=\n", df.head(5)

np_features = df.as_matrix()
np_keys = keys.as_matrix()

features_train, features_test, keys_train, keys_test = train_test_split(np_features, np_keys, test_size=0.33, random_state=42)

print "type(features_train)=", type(features_train), "features_train.shape=", features_train.shape
print "type(features_test)=", type(features_test), "features_test.shape=", features_test.shape
print "type(keys_train)=", type(keys_train), "keys_train.shape=", keys_train.shape
print "type(keys_test)=", type(keys_test), "keys_test.shape=", keys_test.shape

del df
del np_features
del np_keys

start_time = time.time()
clf = ensemble.RandomForestClassifier(n_estimators=200,n_jobs=-1,random_state=0)
print("--- time to execute  ensemble.RandomForestClassifier %s seconds ---" % (time.time() - start_time))
#http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
#n_jobs : integer, optional (default=1)
# The number of jobs to run in parallel for both fit and predict.
# If -1, then the number of jobs is set to the number of cores.
# my I7-4790 has 4 cores.

print "fitting clf.fit"
start_time = time.time()
clf.fit(features_train, keys_train)
print("--- time to execute  clf.fit %s seconds ---" % (time.time() - start_time))

print "predicting"
start_time = time.time()
keys_test_predicted = clf.predict(features_test)
print("--- time to predict %s seconds ---" % (time.time() - start_time))

print "calculating accuracy_score"
start_time = time.time()
score = accuracy_score(keys_test, keys_test_predicted)
print("--- time to calcualte accuracy_score %s seconds ---" % (time.time() - start_time))
print "type(score)=", type(score)

print "accuracy_score(keys_test, keys_test_predicted) = ", score
