import time
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import ensemble
from sklearn.metrics import accuracy_score
import numpy as np

loc_test = "./data/test.csv"
loc_train = "./data/train.csv"

print ("loading test and train data")
start_time = time.time()
df_test = pd.read_csv(loc_test)
print ("type(df_test)=", type(df_test), "df_test.shape=", df_test.shape)

df_train = pd.read_csv(loc_train)
print ("type(df_train)=", type(df_train), "df_train.shape=", df_train.shape)
print("--- time to load data %s seconds ---" % (time.time() - start_time))

print ("df_train.head(5)=\n", df_train.head(5))

print ("df_train column names = ", list(df_train.columns.values) )
print ("df_test column names = ", list(df_test.columns.values) )

feature_cols = [col for col in df_train.columns if col not in ['species', 'id']]
X_train = df_train[feature_cols]
Y_train = df_train['species']
X_test = df_test[feature_cols]
test_ids = df_test['id']

print ("type(X_train)=", type(X_train), "X_train.shape=", X_train.shape)
print ("type(Y_train)=", type(Y_train), "Y_train.shape=", Y_train.shape)

print ("type(X_test)=", type(X_test), "X_test.shape=", X_test.shape)
print ("type(test_ids)=", type(test_ids), "test_ids.shape=", test_ids.shape)

np_x_train = X_train.as_matrix()
np_y_train = Y_train.as_matrix()
np_x_test = X_test.as_matrix()
np_test_ids = test_ids.as_matrix()

print ("type(np_x_train)=", type(np_x_train), "np_x_train.shape=", np_x_train.shape)
print ("type(np_y_train)=", type(np_y_train), "np_y_train.shape=", np_y_train.shape)
print ("type(np_x_test)=", type(np_x_test), "np_x_test.shape=", np_x_test.shape)
print ("type(np_test_ids)=", type(np_test_ids), "np_test_ids.shape=", np_test_ids.shape)

print ("creating ensemble.RandomForestClassifier")
start_time = time.time()
clf = ensemble.RandomForestClassifier(n_estimators=200,n_jobs=-1,random_state=0)
print("--- time to execute  ensemble.RandomForestClassifier %s seconds ---" % (time.time() - start_time))
#http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html


print ("fitting clf.fit")
start_time = time.time()
clf.fit(np_x_train, np_y_train)
print("--- time to execute  clf.fit %s seconds ---" % (time.time() - start_time))


print ("predicting")
start_time = time.time()
np_train_keys_predicted = clf.predict(np_x_train)
print("--- time to predict %s seconds ---" % (time.time() - start_time))

print ("calculating accuracy_score on training data.")
start_time = time.time()
score = accuracy_score(np_y_train, np_train_keys_predicted)
print("--- time to calcualte accuracy_score %s seconds ---" % (time.time() - start_time))
print ("type(score)=", type(score))
print ("accuracy_score(np_y_train, np_train_keys_predicted) = ", score)



