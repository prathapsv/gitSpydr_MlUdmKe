# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:47:37 2020

@author: prathap
"""

# Simple Linear Regression model implementation in Python.
#  E:\SL\ai\udemy\mtrls\codes_frm_ml-crse\Machine Learning A-Z New\Part 2 - Regression\Section 4 - Simple Linear Regression

# Data Pre-Processing Template.

#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#loading dataset
df=pd.read_csv('E:\\SL\\ai\\udemy\\mtrls\\codes_frm_ml-crse\\Machine Learning A-Z New\\Part 2 - Regression\\Section 4 - Simple Linear Regression\\Salary_Data.csv')

X=df.iloc[:,:-1].values
y =df.iloc[:,1].values

# Note: Below commented codes were not required, for the sake of learning the ML algos as the data we get
# will be pre-processed already, but just kept for reference.
# Handling NaN/Null values + Encoding Categorical Features + Creating Dummy variables from categorical features.
"""
# Handling null/Nan values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3]=imputer.transform(X[:, 1:3])

# encoding categorical data..
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

# Encoding Y data
from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)
"""

# Splitting the dataset into two sets: Training Set and Test Set.
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

# Note: Below commented codes were not required, for the sake of learning the ML algos as the data we get
# will be pre-processed already, but just kept for reference.
# Feature Scaling of the data(X axis data/independant features)
"""
# Feature scaling.
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
"""

# Note, we are not doing Feature Scaling for the dataset, bcz the model(sklearn.linear_model's  LinearRegression)
# we are using will take care of Feature Scaling by itself.
# applying Simple Linear Regression model

# Fitting Simple Linear Regression to Training Set.
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

# Predicting the Test set results.
y_pred=regressor.predict(X_test)

# Visualising the Training Set results.
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Salary')
plt.ylabel('Experience')
plt.show();











