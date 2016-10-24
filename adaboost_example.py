"""
https://abcsofds.wordpress.com/2015/10/05/a-is-for-adaboost-classifiers/
"""
import time
import pandas as pd
 
filename_train = './adult_data/adult.data'
filename_test = './adult_data/adult.test'
columns = ['age', 'workclass', 'fnlwgt', 'education', 
           'education-num', 'marital-status', 'occupation', 'relationship',
           'race', 'sex', 'capital-gain', 'capital-loss', 
           'hours-per-week', 'native-country', 'class_label']
 
train = pd.read_csv(filename_train, 
                    sep = ',', 
                    header = None,
                    skipinitialspace=True, 
                    na_values = ['?']).dropna() 
 
test = pd.read_csv(filename_test, 
                   sep = ',', 
                   header = None, 
                   skipinitialspace=True, 
                   na_values = ['?'], 
                   skiprows = [0]).dropna() 
 
train.columns = columns
test.columns = columns

print type(train)
print type(test)
print "train.shape=", train.shape #(30162, 15)
print "test.shape=", test.shape   #(15060, 15)

print "train.head(5)=\n", train.head(5)
print "test.head(5)=\n", test.head(5)

class_label = {'=>50K':1, '<=50K':0, '=>50K.':1, '<=50K.':0, '>50K.' : 1, '>50K' : 1}

workclass = {'Private':0, 'Self-emp-not-inc':1, 'Self-emp-inc':2,
            'Federal-gov':3, 'Local-gov':4, 'State-gov':5,
            'Without-pay':6, 'Never-worked':7}

print type(class_label)

print type(workclass)

#setup dictionaries for converting text data to numerical.
education = {'Bachelors':0, 'Some-college':1, '11th':2,
             'HS-grad':3, 'Prof-school':4, 'Assoc-acdm':5,
             'Assoc-voc':6, '9th':7, '7th-8th':8,
             '12th':9, 'Masters':10, '1st-4th':11,
             '10th':12, 'Doctorate':13, '5th-6th':14, 'Preschool':15}

marital_status = {'Married-civ-spouse':0, 'Divorced':1, 'Never-married':2,
                  'Separated':3, 'Widowed':4, 'Married-spouse-absent':5,
                  'Married-AF-spouse':6}

occupation = {'Tech-support':0, 'Craft-repair':1, 'Other-service':2,
              'Sales':3, 'Exec-managerial':4, 'Prof-specialty':5,
              'Handlers-cleaners':6, 'Machine-op-inspct':7,
              'Adm-clerical':8, 'Farming-fishing':9, 'Transport-moving':10,
              'Priv-house-serv':11, 'Protective-serv':12, 'Armed-Forces':13}

relationship = {'Wife':0, 'Own-child':1, 'Husband':2,
                'Not-in-family':3, 'Other-relative':4, 'Unmarried':5}

race = {'White':0, 'Asian-Pac-Islander':1, 'Amer-Indian-Eskimo':2,
        'Other':3, 'Black':4}

sex = {'Female':0, 'Male':1}

native_country = {'United-States':0, 'Cambodia':1, 'England':2,
                  'Puerto-Rico':3, 'Canada':4, 'Germany':5,
                  'Outlying-US(Guam-USVI-etc)':6, 'India':7, 'Japan':8,
                  'Greece':9, 'South':10, 'China':11,
                  'Cuba':12, 'Iran':13, 'Honduras':14,
                  'Philippines':15, 'Italy':16, 'Poland':17,
                  'Jamaica':18, 'Vietnam':19, 'Mexico':20,
                  'Portugal':21, 'Ireland':22, 'France':23,
                  'Dominican-Republic':24, 'Laos':25, 'Ecuador':26,
                  'Taiwan':27, 'Haiti':28, 'Columbia':29,
                  'Hungary':30, 'Guatemala':31, 'Nicaragua':32,
                  'Scotland':33, 'Thailand':34, 'Yugoslavia':35,
                  'El-Salvador':36, 'Trinadad&Tobago':37, 'Peru':38,
                  'Hong':39, 'Holand-Netherlands':40}

# use pandas and the dicts created to replace each string attribute value with its corresponding number.
train = train.replace({'workclass':workclass,
                       'education': education,
                       'marital-status':marital_status,
                       'occupation': occupation,
                       'relationship': relationship,
                       'race': race,
                       'sex': sex,
                       'native-country': native_country,
                       'class_label': class_label})

#
print "train.head(5)=\n", train.head(5)

test = test.replace({'workclass':workclass,
                     'education': education,
                     'marital-status':marital_status,
                     'occupation': occupation,
                     'relationship': relationship,
                     'race': race,
                     'sex': sex,
                     'native-country': native_country,
                     'class_label': class_label})

print "test.head(5)=\n", test.head(5)

import numpy as np

#convert to a numpy array so the data can be processed by scikit-learn packages.
train_np = pd.DataFrame.as_matrix(train)
test_np = pd.DataFrame.as_matrix(test)

#print "train_np"
print "type(train_np)=", type(train_np)
print "type(test_np)=", type(test_np)
print "train_np.shape=", train_np.shape #NB: same shape as the original dataframe
print "test_np.shape=", test_np.shape   #NB: same shape as the original dataframe
print "train_np.size = ", train_np.size #452,430, nb 452,430 = 30162 x 15, train.shape = (30162, 15)
print "test_np.size=", test_np.size     #225,900, nb 225,900 = 15060 x 15, test.shape  = (15060, 15)
#NBB: numpy.ndarray.size is misleading.

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
#http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
#http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

print "initialising AdaBoostClassifier"
start_time = time.time()
ada = AdaBoostClassifier(base_estimator = RandomForestClassifier(n_estimators = 20, criterion = 'entropy'), algorithm = 'SAMME.R')
print("--- time to initialise AdaBoostClassifier %s seconds ---" % (time.time() - start_time))

#AdaBoostClassifier : algorithm : {'SAMME', 'SAMME.R'}, optional (default='SAMME.R')
#n_estimators : integer, optional (default=10)  The number of trees in the forest.
#criterion : string, optional (default="gini") Supported criteria are "gini" for the Gini impurity and "entropy" for the information gain
#algorithm


#now fit model using data from train_np.
# feature data = All rows except last column, predicted data = last column only.
start_time = time.time()
ada.fit(train_np[:, :-1], train_np[:, -1])
print("--- time to fit AdaBoostClassifier %s seconds ---" % (time.time() - start_time))
#train_np[:, :-1] = all rows & all columns except last column.
#train_np[:, -1] = all rows & all last column only.


print "type(ada)=", type(ada)

#now predict values using the test dataset. [all rows except the last column]
start_time = time.time()
predictions = ada.predict(test_np[:,:-1])
print("--- time to predict AdaBoostClassifier %s seconds ---" % (time.time() - start_time))

print "type(predictions)=", type(predictions)

#now score the accuracy, use all rows except last column from input data, compare to last column.
start_time = time.time()
score = ada.score(test_np[:,:-1], test_np[:,-1])
print("--- time to score AdaBoostClassifier %s seconds ---" % (time.time() - start_time))

print "test_np column names = (test.columns) = ", list(test.columns)
#train_np[:, :-1] takes all rows and all columns except the last of train_np (all attributes except class label)
#train_np[:, -1] takes all rows of only the last column

print "type(score)=", type(score)
print 'Accuracy:', "%.2f" %(score*100)+'%'

#Accuracy: 83.63%

print "test_np[:,-1] = last column name: ", list(test.columns)[-1]

print 'Number of people earning =>50K: ', sum(test_np[:,-1])
print 'Number of people earning <=50K: ', len(test_np[:,-1]) - sum(test_np[:,-1])

ada = AdaBoostClassifier()

from sklearn.metrics import confusion_matrix
#http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

start_time = time.time()
cm = confusion_matrix(test_np[:,-1], predictions )
print("--- time to calc confusion_matrix %s seconds ---" % (time.time() - start_time))

import matplotlib.pyplot as plt
np.set_printoptions(precision=2)
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.set_printoptions.html

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print "type(cm_normalized)=", type(cm_normalized)
print "cm_normalized=", cm_normalized
print "True Positive = ", cm_normalized[0,0]
print "False Negative = ", cm_normalized[0,1]
print "False Positive = ", cm_normalized[1,0]
print "True Negative = ", cm_normalized[1,1]

plt.figure()
plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Normalized Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['<=50K', '=>50K'], rotation=45)
plt.yticks(tick_marks, ['<=50K', '=>50K'])
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print "type(cm)", type(cm)
print "cm=", cm

