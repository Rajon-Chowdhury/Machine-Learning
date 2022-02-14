#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from apyori import  apriori


# In[2]:


data = pd.read_csv('depression.csv')
data.head()


# In[3]:


Total = len(data)
records = []
for i in range(0,Total):
    records.append([str(data.values[i,j]) for j in range(0,20)])

association_rules = apriori(records, min_support=0.069, min_confidence=0.5, min_lift=3, min_length=2)
rules = association_rules
association_results = list(association_rules)
print(len(association_results))


# In[4]:


results = []
i = 0

for dataitem in association_results:
    i += 1
    pair = dataitem[0]
    items = [x for x in pair]
    print(f'Rule [{i}]: antecedent: {items[0]} -> consequents: {items[1]}')
    print("Support: " + str(dataitem[1]))
    print("Confidence: " + str(dataitem[2][0][2]))
    print('Lift: ' + str(dataitem[2][0][3]))


# In[ ]:





# In[ ]:




