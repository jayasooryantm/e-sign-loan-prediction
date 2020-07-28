# -*- coding: utf-8 -*-
"""
Title: Predicting the Likelihood of E-Signing a Loan Based on Financial
       History
Created on Tue Jul 28 17:38:38 2020

@author: Jayasooryan TM
"""

# importing necessary libraries
import pandas as pd
import random
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

random.seed(100)

# Loading the data

dataset = pd.read_csv(r'P39-Financial-Data.csv')

# Feature Engineering

dataset = dataset.drop(['months_employed'],axis=1)
dataset['personal_account_months'] = (dataset['personal_account_m']
                                      + (dataset['personal_account_y'] * 12))
dataset[['personal_account_months','personal_account_m','personal_account_y']]
dataset.drop(columns=['personal_account_m','personal_account_y'], inplace=True)

# One-hot encoding

dummies = pd.get_dummies(dataset['pay_schedule'], drop_first=True)
dataset = dataset.drop(columns=['pay_schedule'])
dataset = pd.concat([dataset,dummies], axis=1)


# removing extra columns

target = dataset['e_signed']
user = dataset['entry_id']
features = dataset.drop(columns=['e_signed','entry_id'])


# train test split

x_train, x_test, y_train, y_test = train_test_split(features, target,
                                                    random_state=0,
                                                    test_size=0.2)

# scaling the features

# copying the DF schema
x_trainScaled = x_train.copy()
x_testScaled = x_test.copy()

scaler = StandardScaler()
x_trainScaled[x_train.columns] = scaler.fit_transform(x_train)
x_testScaled[x_test.columns] = scaler.transform(x_test)

### Model Building

# imorting metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


## Comparing the models

# logistic regression
from sklearn.linear_model import LogisticRegression
log_classifier = LogisticRegression(random_state=0, penalty='l2')
log_classifier.fit(x_trainScaled, y_train)

# predicitng the test samples
y_pred = log_classifier.predict(x_testScaled)

acc = accuracy_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)

# generating classification report

report = pd.DataFrame([['Logistic Regression (ridge)',acc,rec,prec,f1]],
                      columns=['Model','Accuracy','Recall','Precision','F1 score'])
print(report)

# SVM (linear)
from sklearn.svm import SVC
classifier = SVC(random_state=0, kernel='linear')
classifier.fit(x_trainScaled, y_train)

# predicitng the test samples
y_pred = classifier.predict(x_testScaled)

acc = accuracy_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)

# generating classification report

report = report.append(pd.DataFrame([['SVM (Linear)',acc,rec,prec,f1]],
                      columns=['Model','Accuracy','Recall','Precision','F1 score']), ignore_index=True)
print(report)

# SVM (RBF)
from sklearn.svm import SVC
classifier = SVC(random_state=0, kernel='rbf')
classifier.fit(x_trainScaled, y_train)

# predicitng the test samples
y_pred = classifier.predict(x_testScaled)

acc = accuracy_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)

# generating classification report

report = report.append(pd.DataFrame([['SVM (rbf)',acc,rec,prec,f1]],
                      columns=['Model','Accuracy','Recall','Precision','F1 score']), ignore_index=True)
print(report)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(random_state=0,
                                    criterion='entropy',
                                    n_estimators=100)
classifier.fit(x_trainScaled, y_train)

# predicitng the test samples
y_pred = classifier.predict(x_testScaled)

acc = accuracy_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)

# generating classification report

report = report.append(pd.DataFrame([['RandomForestClassifier',acc,rec,prec,f1]],
                      columns=['Model','Accuracy','Recall','Precision','F1 score']), ignore_index=True)
print(report)



# finding the cross validation score - best score

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X=x_trainScaled, y=y_train,
                             cv=10)
print('Random Forest Classifier accuracy: %0.2f (+/- %0.2f)' % (accuracies.mean(), accuracies.std()*2))


# Applying cross validation
from sklearn.model_selection import GridSearchCV

params= {'max_depth':[3,None],
         'max_features':[1,5,10],
         'min_samples_split':[2,5,10],
         'min_samples_leaf':[1,5,10],
         'bootstrap':[True,False],
         'criterion':["entropy"]}

grid_search = GridSearchCV(estimator= classifier,
                           param_grid=params,
                           scoring='accuracy',
                           cv=10,
                           n_jobs=-1)

t0 = time.time()
grid_search = grid_search.fit(x_trainScaled,y_train)
t1 = time.time()
print('Time consumption: %0.2f' % (t1-t0))

''' Best Estimator : 
    RandomForestClassifier(bootstrap: True,
                                    criterion: 'entropy',
                                    max_depth: None,
                                    max_features: 10,
                                    min_samples_leaf: 5,
                                    min_samples_split: 2
                                    random_state=0)
'''
classifier = RandomForestClassifier(bootstrap= True,
                                    criterion= 'entropy',
                                    max_depth= None,
                                    max_features= 10,
                                    min_samples_leaf= 5,
                                    min_samples_split= 2,
                                    random_state=0)

classifier.fit(x_trainScaled, y_train)

# predicitng the test samples
y_pred = classifier.predict(x_testScaled)

acc = accuracy_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)

# generating classification report

report = report.append(pd.DataFrame([['RandomForestClassifier (Best Estimator)',acc,rec,prec,f1]],
                      columns=['Model','Accuracy','Recall','Precision','F1 score']), ignore_index=True)

print(report)



