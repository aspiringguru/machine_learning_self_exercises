"""
case study example
https://www.analyticsvidhya.com/blog/2015/06/tuning-random-forest-model/

does not work, need to convert string data into number for scikit-learn packages to operate with numpy array of numbers.

"""

from sklearn.ensemble import RandomForestRegressor
#http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

from sklearn.metrics import roc_auc_score
#http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
#Compute Area Under the Curve (AUC) from prediction scores

import pandas as pd
x = pd.read_csv("./data/train.csv")
print ("type(x)", type(x), "x.shape=", x.shape)
print ("column names = ", list(x.columns.values) )
print ( "x.head(10)=\n", x.head(10) )


y = x.pop("Survived")
print ("column names = ", list(x.columns.values) )

print ("type(y)", type(y), "y.shape=", y.shape)

model =  RandomForestRegressor(n_estimators = 100 , oob_score = True, random_state = 42)

model.fit(x['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],y)

#print ("AUC - ROC : ", roc_auc_score(y,model.oob_prediction) )