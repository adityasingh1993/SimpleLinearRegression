# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 22:02:37 2017

@author: Aditya
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Loading Data
dataset=pd.read_csv('Salary_Data.csv')
print (dataset)
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

            
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------             
#Missing value data insertion, since our datasheet does'nt have any missing values we wont do it.
#But we will keep the code in comment



#from sklearn.preprocessing import Imputer
#imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)  
#X=imputer.fit_transform(X[:,0]) 


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


#As we know encoding is required when we caterogical data, So here we dont require encoding, but we will write code


#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelEncoder_X=LabelEncoder()
#X[:,0]=labelEncoder_X.fit_transform(X[:,0])
#oneHotEncoder_X=OneHotEncoder(caterogical_features=[0])
#X=oneHotEncoder_X.fit_transform(X).toarray()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Splitting The Data

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#Feature Selection
"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X)
X_test=sc_X.transform(X)"""

                     
#Simple Linear Regression

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)
#print(y_pred)
My_pred=regressor.predict(40)

plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience(ONTrain)')
plt.xlabel('Years Of Exp')
plt.ylabel('Salary')
plt.show()
                           
