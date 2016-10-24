"""
This code was published on github 25/10/2016.
https://github.com/aspiringguru/machine_learning_self_exercises

https://www.kaggle.com/triskelion/forest-cover-type-prediction/first-try-with-random-forests/code
https://archive.ics.uci.edu/ml/datasets/Covertype

Predict the forest cover type (the predominant kind of tree cover) from strictly cartographic variables
(as opposed to remotely sensed data).

Note: this code demonstrates a 'cheat' for the kaggle comp.
I wrote this after wondering how other submissions were achieving a perfect score, when the kaggle problem
definition nominates a test dataset representing 2% of the total data.
I scored 75% in my previous attempt - which used sample code from the kaggle demo.
(see KaggleForestCoverRandomForest1.py in this github repo)

This Kaggle comp is closed, and no kaggle points, so it's really just an educational/demo exercise.

Pitchforks on sale at pitchforkemporium.
https://www.reddit.com/r/pitchforkemporium/

TLDR: overfitting is bad.

"""

import time
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import ensemble
from sklearn.metrics import accuracy_score
import numpy as np

raw_data = "./covtype/covtype.data"
loc_test = "./covtype/test.csv"
loc_train = "./covtype/train.csv"
loc_submission = "./covtype/kaggle.forest-cover-type-prediction.AspiringGuru.csv"

df_test = pd.read_csv(loc_test)
df_train = pd.read_csv(loc_train)

print "type(df_test)=", type(df_test), "df_test.shape=", df_test.shape
feature_cols = [col for col in df_train.columns if col not in ['Cover_Type', 'Id']]
X_test = df_test[feature_cols]
test_ids = df_test['Id']

del df_train
del df_test


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


print "type(np_features)=", type(np_features), "np_features.shape=", np_features.shape
print "type(np_keys)=", type(np_keys), "np_keys.shape=", np_keys.shape

del df

print "creating ensemble.RandomForestClassifier"
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
clf.fit(np_features, np_keys)
print("--- time to execute  clf.fit %s seconds ---" % (time.time() - start_time))

print "predicting"
start_time = time.time()
np_keys_predicted = clf.predict(np_features)
print("--- time to predict %s seconds ---" % (time.time() - start_time))

print "calculating accuracy_score"
start_time = time.time()
score = accuracy_score(np_keys, np_keys_predicted)
print("--- time to calcualte accuracy_score %s seconds ---" % (time.time() - start_time))
print "type(score)=", type(score)

print "accuracy_score(np_keys, np_keys_predicted) = ", score


print "clf.predicting & writing to file"
start_time = time.time()
with open(loc_submission, "w") as outfile:
    outfile.write("Id,Cover_Type\n")
    for e, val in enumerate(list(clf.predict(X_test))):
        outfile.write("%s,%s\n" % (test_ids[e], val))
print("--- time to clf.predict & write to file %s seconds ---" % (time.time() - start_time))
