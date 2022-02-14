#!/usr/bin/env python
# coding: utf-8

# In[160]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[161]:


df = pd.read_csv("depression.csv")

columns = ["Timestamp","Year","Feelings","Scale","Gender","Age","Location","Relationship","Finance","Adjustment","Understanding","Study_pressure","Result","Living","Support","Social_media","InferiorityComplex","Meal","Sick","Recreation","Sleep"]
df.columns = columns
df.head()


# In[162]:


sns.boxplot(x="Scale",y="Feelings",data=df)


# In[163]:


sns.boxplot(x="Sleep",y="Feelings",data=df)


# In[164]:


sns.swarmplot(x=df['Feelings'],y=df['Scale'])


# In[165]:



df.drop(['Meal','Timestamp','Year','Age','Gender','Location','Social_media','Support','Relationship','InferiorityComplex'],axis=1,inplace=True)


# In[166]:


import category_encoders as ce
encoder = ce.OrdinalEncoder(cols=['Finance','Understanding','Study_pressure','Result','Living','Sick','Recreation'])

df = encoder.fit_transform(df)


# In[167]:


df['Feelings'] = df['Feelings'].map({'Very good': 0, 'Good': 1, 'Normal': 2, "Bad": 3, 'Very bad' : 4})


# In[168]:


plt.figure(figsize=(15,15))
p=sns.heatmap(df.corr(), annot=True,cmap='coolwarm',center=0) 


# In[169]:


data1 = pd.get_dummies(df['Finance'],drop_first=True)
data2 = pd.get_dummies(df['Understanding'],drop_first=True)
data3 = pd.get_dummies(df['Study_pressure'],drop_first=True)
data4 = pd.get_dummies(df['Result'],drop_first=True)
data5 = pd.get_dummies(df['Living'],drop_first=True)
data6 = pd.get_dummies(df['Sick'],drop_first=True)
data7 = pd.get_dummies(df['Recreation'],drop_first=True)


# In[170]:


data1 = data1.rename(columns={1 :'Finance_Yes', 2 : 'Finance_No'})
data2 = data2.rename(columns={1 :'Understanding_Good', 2 : 'Understanding_Normal',3:'Understanding_Bad'})
data3 = data3.rename(columns={1 :'Study_pressure_No', 2 : 'Study_pressure_Yes'})
data4 = data4.rename(columns={1 :'Result_No', 2 : 'Result_Yes'})
data5 = data5.rename(columns={1 :'Living_No', 2 : 'Living_Yes'})
data6 = data6.rename(columns={1 :'Sick_No', 2 : 'Sick_Yes'})
data7 = data7.rename(columns={1 :'Recreation_No', 2 : 'Recreation_Yes'})


# In[171]:


df = pd.concat([df, status1], axis = 1)
df = pd.concat([df, status2], axis = 1)
df = pd.concat([df, status3], axis = 1)
df = pd.concat([df, status4], axis = 1)
df = pd.concat([df, status5], axis = 1)
df = pd.concat([df, status6], axis = 1)
df = pd.concat([df, status7], axis = 1)


# In[172]:


df.drop(['Finance'], axis = 1, inplace = True)
df.drop(['Understanding'], axis = 1, inplace = True)
df.drop(['Study_pressure'], axis = 1, inplace = True)
df.drop(['Result'], axis = 1, inplace = True)
df.drop(['Living'], axis = 1, inplace = True)
df.drop(['Sick'], axis = 1, inplace = True)
df.drop(['Recreation'], axis = 1, inplace = True)


# In[173]:


X = df.drop(['Feelings'], axis=1)
y = df['Feelings']


# In[174]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state =42)


# In[175]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[184]:


from sklearn.neural_network import MLPClassifier
MLP = MLPClassifier(hidden_layer_sizes=(16,32,64),activation='relu',solver='adam',max_iter=1000)
MLP.fit(X_train,y_train)


# In[185]:


predictions = MLP.predict(X_test)
predictions


# In[ ]:





# In[186]:


from sklearn.metrics import accuracy_score,classification_report

print(classification_report(y_test,predictions))


# In[ ]:





# In[187]:


accuracy_score(y_test,predictions)*100


# In[ ]:




