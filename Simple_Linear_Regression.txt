
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

dataset=pd.read_csv('CNNdata.csv')
X = dataset.iloc[:,:-1].values
y= dataset.iloc[:,1].values


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size= 0.3, random_state=0)


# fitting Simple linear Regression to the traing set 
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train, y_train)

#predict the test set results
y_pred= regressor.predict(X_test)
'''
#Visualizing trainig results 
plt.scatter(X_train,y_train , color= 'red')
plt.plot(X_train, regressor.predict(X_train), color='green')
plt.title('Salary vs Experiance(Traing Set)')
plt.xlabel('Years of Experiance')
plt.ylabel('Salary')
plt.show()'''
#VIsualizing test results 
plt.scatter(X_test,y_test , color= 'red')
plt.plot(X_train, regressor.predict(X_train), color='green')
plt.title('Salary vs Experiance(Test Set)')
plt.xlabel('Years of Experiance')
plt.ylabel('Salary')
plt.show()