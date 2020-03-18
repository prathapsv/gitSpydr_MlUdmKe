#Data Preprocessing

#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#loading dataset
df=pd.read_csv('E:\\SL\\ai\\udemy\\mtrls\\P14-Data-Preprocessing\\Data_Preprocessing\\Data.csv')

ipndtFeatures_X=df.iloc[:,:-1].values
dpndtFeature_Y =df.iloc[:,3].values 

#%% region deprecated
 # manage missing values
#from sklearn.preprocessing import Imputer
## using mean strategy, meaning it replaces the NaN values with mean of the column.
#imputer=Imputer(missing_values ='NaN',strategy='mean',axis=0)
#imputer=imputer.fit(ipndtFeatures_X[:,1:3])# age and salary columns.
#ipndtFeatures_X[:,1:3]=imputer.transform(ipndtFeatures_X[:,1:3])
#%%

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
imputer = imputer.fit(ipndtFeatures_X[:, 1:3])
ipndtFeatures_X[:, 1:3]=imputer.transform(ipndtFeatures_X[:, 1:3])

# encoding categorical data..
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
ipndtFeatures_X = np.array(ct.fit_transform(ipndtFeatures_X), dtype=np.float)

# Encoding Y data
from sklearn.preprocessing import LabelEncoder
dpndtFeature_Y = LabelEncoder().fit_transform(dpndtFeature_Y)

# encoding country column data.
#%% region deprecated
#from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#lblEncoder_Country_X=LabelEncoder()
#ipndtFeatures_X[:,0]=lblEncoder_Country_X.fit_transform(ipndtFeatures_X[:,0])
#
##encoding purchased column data.
#lblEncoder_Purchased_Y=LabelEncoder()
#dpndtFeature_Y=lblEncoder_Purchased_Y.fit_transform(dpndtFeature_Y)

# dummy variables encoding for country column after converting it into numerical data column from 
# categorical data column using OneHotEncoder
#oneHotEncoder_Country_X=OneHotEncoder(categorical_features=[0])
#ipndtFeatures_X=oneHotEncoder_Country_X.fit_transform(ipndtFeatures_X).toarray()
#%%

# Splitting the dataset into two sets: Training Set and Test Set.
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(ipndtFeatures_X,dpndtFeature_Y,test_size=0.2
                                                   ,random_state=0)

# Feature scaling.
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)








