# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the needed packages.
2. Assigning hours To X and Scores to Y.
3. Plot the scatter plot.
4. Use mse,rmse,mae formmula to find the values.
    

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Sarish Varshan V
RegisterNumber:  212223230196
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv("C:/Users/admin/Downloads/student_scores (1).csv")
df.head()
df.tail()
X=df.iloc[:,:-1].values
print(X)
Y=df.iloc[:,-1].values
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

print(Y_pred)
print(Y_test)
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="orange")
plt.plot(X_test,regressor.predict(X_test),color="red")
plt.title("Hours vs scores(Test Data Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE : ",rmse)
```
### output:
![image](https://github.com/sarishvarshan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152167665/02b3be17-d981-4861-992b-f03fb1a92d29)

![image](https://github.com/sarishvarshan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152167665/c2552d24-ed05-464b-97ec-96e53a24cf0b)

![image](https://github.com/sarishvarshan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152167665/4ef73f76-c905-4b96-96b5-3d10bbad58f7)

![image](https://github.com/sarishvarshan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152167665/90ccb525-1ed7-45d2-829b-ddcc25e4eef2)

![image](https://github.com/sarishvarshan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152167665/62e88d1f-683f-4428-bc63-b0eeda038ae1)

![image](https://github.com/sarishvarshan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152167665/6ef1f2e9-7577-4dca-bb29-d8155a0238a9)

![image](https://github.com/sarishvarshan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152167665/2a116f34-e259-42ce-a306-52aeb16adfc5)

![image](https://github.com/sarishvarshan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152167665/7b8ecade-a71c-4d69-964b-997b47201585)

![image](https://github.com/sarishvarshan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152167665/0dd3a3cc-c863-4548-b51b-36376a600956)





## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming .
