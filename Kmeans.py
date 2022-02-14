#!/usr/bin/env python
# coding: utf-8

# In[48]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as splt
import seaborn
get_ipython().run_line_magic('matplotlib', 'inline')

class KMeans:
    def __init__(self, clusters_Num, n_iteration=10000):
        self.clusters_Num = clusters_Num
        self.n_iteration = n_iteration
        
    def distance_Calculation(self, center, instance):
        distance = np.linalg.norm(center - instance)
        return distance   
         
    def create_random_cluster_centers(self, data):
        centers = data.sample(n=self.clusters_Num)
        return centers.to_numpy()

    def assign_cluster(self, data, centers):
        clusters = [[] for _ in range(self.clusters_Num)]
        for instance in data:
            clusters[self.findOut__nearest_cluster(centers, instance)].append(instance)
        return clusters

    def update_centers(self, centers, clusters):
        new_centers = []
        for center, cluster in zip(centers, clusters):
            if len(cluster) == 1:
                continue
            center = np.mean(cluster, axis=0)
            new_centers.append(center)
        return np.array(new_centers)


    def findOut__nearest_cluster(self, centers, instance):
        min_distance = 1000
        cluster_no = 0
        for i, center in enumerate(centers):
            distance = self.distance_Calculation(center, instance)
            if distance < min_distance:
                min_distance = distance
                cluster_no = i
        return cluster_no

    def fit(self, data):
        self.n_features = data.shape[1]
        centers = self.create_random_cluster_centers(data)
        data_origin=data.copy()
        data = data.to_numpy()
        for i in range(self.n_iteration):
            clusters = self.assign_cluster(data, centers)
            new_centers = self.update_centers(centers, clusters)
            if (centers == new_centers).all():
                break
            centers = new_centers
        self.n_iteration = i + 1
        self.centers = centers
        self.labels = self.predict(data_origin)
    def predict(self, data):
        if isinstance(data, pd.Series):
            data = data.to_frame().T
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data.reshape(-1, self.n_features))
        clusters = []
        for row in data.iterrows():
            row = row[1]
            cluster_no = self.findOut__nearest_cluster(self.centers, row)
            clusters.append(cluster_no)
        return np.array(clusters)


# In[49]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[50]:


data = pd.read_csv("depression.csv")
df=data
columns = ["Timestamp","Year","Mood","Scale","Gender","Age","Residence","Relationship","FinancialState","Adjustment","RelationWithFamily","Pressure","AcademicResult","LivingPlace","SupportedBy","SocialMedia","InferiorityComplex","MealSatisfaction","Health","OtherPositiveActivity","SleepTime"]
data.columns = columns
df.columns = columns
data = data.sample(frac=1).reset_index(drop=True)
data.drop(["Mood","Timestamp","Scale"], axis=1, inplace=True)
columns = ["Year","Gender","Age","Residence","Relationship","FinancialState","Adjustment","RelationWithFamily","Pressure","AcademicResult","LivingPlace","SupportedBy","SocialMedia","InferiorityComplex","MealSatisfaction","Health","OtherPositiveActivity","SleepTime"]
data_Enc = pd.get_dummies(data, columns=columns)


# In[51]:


from sklearn.preprocessing import LabelEncoder
df = df.apply(LabelEncoder().fit_transform)
seaborn.pairplot(df[['Age','FinancialState']])


# In[52]:


seaborn.pairplot(df[['Adjustment','AcademicResult']])


# In[53]:


kmeans = KMeans(clusters_Num=5)
kmeans.fit(data_Enc)

print("Labels for the data  :{}".format(kmeans.labels))

print("Centers :{}".format(kmeans.centers))


# In[ ]:





# In[ ]:





# In[54]:



from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(data_Enc)


# In[55]:


print("Labels for the data  :{}".format(kmeans.labels_))

print("Centers :{}".format(kmeans.cluster_centers_))


# In[ ]:





# In[ ]:





# In[ ]:




