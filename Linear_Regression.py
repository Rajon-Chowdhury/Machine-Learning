#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import numpy as np

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import mean_squared_error

class LinearRegression:
    def __init__(self, max_iteration = 10000, learning_rate = 0.0002, max_mse = None):
        self.max_iteration=max_iteration
        self.max_mse = max_mse
        self.learning_rate = learning_rate
        return 

    def fit(self, X, Y):
        if isinstance(X,pd.Series):
            X = X.to_frame()
        Y = Y.to_frame()
        self.n = len(X)
        self.coeff = [0 for _ in range(len(X.columns))]
        self.intercept = 0
        self.mse=[]
        self.n_iteration = 0
        X = X.values
        Y = Y.values
        for _ in range(self.max_iteration):
            y_pred = np.sum(X*self.coeff,axis=1) + self.intercept
            y_pred = y_pred.reshape(self.n,1)
            current_mse = np.square(np.subtract(Y,y_pred)).mean()
            self.mse.append(current_mse)
            # dw = -2x(yi-y_pred) 
            # db = -2*(y-y_pred)
            Dw = -2*(X*(Y-y_pred).reshape(self.n,1)).mean(axis=0)
            Db = -2*(Y-y_pred).mean()
            self.coeff = self.coeff - Dw*self.learning_rate
            self.intercept = self.intercept - Db*self.learning_rate
            self.n_iteration = self.n_iteration + 1
           

    def predict(self,X):
        if isinstance(X,list):
            X = [X]
        if isinstance(X,pd.Series):
            X = X.to_frame()
        if isinstance(X,pd.DataFrame):
            X = X.values
        n = len(X)
        return (np.sum(X*self.coeff,axis=1) + self.intercept).reshape(n,1)


# In[41]:


if __name__ == "__main__":
    # Imports
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets


    
 


# In[42]:


data = pd.read_csv("depression.csv")
columns = ["Timestamp","Year","Mood","Scale","Gender","Age","Residence","Relationship","FinancialState","Adjustment","RelationWithFamily","Pressure","AcademicResult","LivingPlace","SupportedBy","SocialMedia","InferiorityComplex","MealSatisfaction","Health","OtherPositiveActivity","SleepTime"]
data.columns = columns
data = data.sample(frac=1).reset_index(drop=True)

Scale = data["Scale"]

data.drop(["Scale","Timestamp"], axis=1, inplace=True)
columns = ["Year","Mood","Gender","Age","Residence","Relationship","FinancialState","Adjustment","RelationWithFamily","Pressure","AcademicResult","LivingPlace","SupportedBy","SocialMedia","InferiorityComplex","MealSatisfaction","Health","OtherPositiveActivity","SleepTime"]
df_Enc = pd.get_dummies(data, columns=columns)
x_train, x_test, y_train, y_test= train_test_split(df_Enc, Scale, test_size= 0.2, random_state=1)


# In[43]:


regressor = LinearRegression(max_iteration=10000,learning_rate=0.002)
regressor.fit(x_train, y_train)


# In[44]:


y_pred = regressor.predict(x_train)

r2Train = r2_score(y_train, y_pred)
print("Training R2 Score: {}".format(r2Train))

train_mse = mean_squared_error(y_train, y_pred)
print("Training mse Score: {}".format(train_mse))


# In[45]:


y_pred = regressor.predict(x_test)

r2Test = r2_score(y_test, y_pred)
print("Testing r2 Score : {}".format(r2Test))
test_mse = mean_squared_error(y_test,y_pred)
print("Testing Mse Score : {}".format(test_mse))


# In[46]:


print(regressor.coeff)
print(regressor.intercept)


# In[47]:


from sklearn.linear_model import LinearRegression


# In[48]:


regressor = LinearRegression()
regressor.fit(x_train, y_train)


# In[49]:


y_pred = regressor.predict(x_train)


# In[50]:


r2Train = r2_score(y_train, y_pred)
print("Training R2 Score: {}".format(r2Train))

train_mse = mean_squared_error(y_train, y_pred)
print("Training mse Score: {}".format(train_mse))


# In[51]:


y_pred = regressor.predict(x_test)

r2Test = r2_score(y_test, y_pred)
print("Testing r2 Score: {}".format(r2Test))
test_mse = mean_squared_error(y_test,y_pred)
print("Testing Mse Score: {}".format(test_mse))


# In[ ]:





# In[ ]:




