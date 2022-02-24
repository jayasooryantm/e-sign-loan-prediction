# e-sign-loan-prediction

### Table of Contents

* Overview
* Motivation
* Technical Aspect
* Requirements
* Run
* Technologies Used
* Credits

### Overview

A RandomForestClassifier model to predict the likelihood of e-Signing a loan based on financial data. This project explores one of the best use-cases in the finance industry. Dataset was tried on different algorithms like LogisticRegression, Support Vector Machine Classifier, and RandomForestClassifier. RandomForestClassifier performs well in this dataset and got an accuracy score of 63.42%.

### Technical Aspect

The machine learning models tested in this dataset are given below with their metrics.

 #|                                  Model  | Accuracy |   Recall | Precision | F1 score
--|-----------------------------------------|----------|----------|-----------|----------  
0 |             Logistic Regression (ridge) | 0.562256 | 0.705913 |  0.576207 | 0.634499 
1 |                            SVM (linear) | 0.568677 | 0.735477 |  0.578068 | 0.647341 
2 |                               SVM (rbf) | 0.597432 | 0.692946 |  0.611162 | 0.649490 
3 |          RandomForestClassifier (n=100) | 0.626745 | 0.681535 |  0.645066 | 0.662799 
4 | RandomForestClassifier (Best Estimator) | 0.634283 | 0.708506 |  0.646168 | 0.675903 


### Requirement

* Python (>=3.7) should be installed on your computer.
* Jupyter notebook or any other code editor should be required to open the code.
* You should install the necessary libraries from the requirement.txt file

### Run

* Load the pickle model using the below code: (x_value and y_value should be replaced with proper data values)
  
```
import pickle
model = pickle.load('.pkl','rb')
model.predict(x_value,y_value)
```

### Technologies Used

* Python
* Spyder (Code Editor)
* Sklearn

### Credits

* Mentor - SuperDataScience Team
* Dataset - https://www.superdatascience.com/
