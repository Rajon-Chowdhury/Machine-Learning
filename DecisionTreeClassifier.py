
import pandas as pd
class DecisionTree:

    class ExpandNode:
        def __init__(self, column,value):
            self.column=column
            self.value=value

    class Node:
        def __init__(self,question,tNode,fNode,leafNode,prediction):
            self.question = question
            self.leafNode = leafNode
            self.tNode = tNode
            self.fNode = fNode
            self.prediction = prediction

    def gini(self,data):
        freCounts = self.totalnoOfClass(data)
        impurity = 1
        for label in freCounts:
            prob_of_label = freCounts[label] / float(len(data))
            impurity -= prob_of_label**2
        return impurity

    def information_Gain(self,left, right, current_uncertainty):
        p = float(len(left)) / (len(left) + len(right))
        return current_uncertainty - p * self.gini(left) - (1 - p) * self.gini(right)

    def find_feature(self,data):
        gain = 0
        question = None
        current_uncertainty = self.gini(data)
        for col in data.drop("Level",axis=1):
            values=data[col].unique()
            for val in values:
                q = self.ExpandNode(col,val)
                trueBranch,falseBranch=self.branchTree(q,data)
                if len(trueBranch)==0 or len(falseBranch)==0:
                    continue
                g = self.information_Gain(trueBranch, falseBranch, current_uncertainty)
                if g >= gain:
                    gain, question = g, q
        return gain,question
    def fit(self,x_train,y_train):
        data=x_train
        data["Level"]=y_train
        gain, question=self.find_feature(data)
        leafNode=False
        predictions=None
        tNode=None

        fNode=None
        if gain==0:
            leafNode = True
            predictions = self.totalnoOfClass(data)
        else:
            trueBranch,falseBranch=self.branchTree(question,data)
            tNode = self.train(trueBranch)
            fNode = self.train(falseBranch)
        self.rootNode=self.Node(question,tNode,fNode,leafNode,predict

    def train(self,data):
        gain, question=self.find_feature(data)
        leafNode=False
        predictions=None
        tNode=None
        fNode=None
        if gain==0:
            leafNode = True
            predictions = self.totalnoOfClass(data)
        else:
            trueBranch,falseBranch=self.branchTree(question,data)
            tNode = self.train(trueBranch)
            fNode = self.train(falseBranch)
        return self.Node(question,tNode,fNode,leafNode,predictions)

    def totalnoOfClass(self,data):
        p= data.groupby("Level")["Level"].count().to_dict()
        for key in p.keys():
            p[key]=(p[key]/len(data))
        return p

    def branchTree(self,question,data):
        trueBranch = data[data[question.column]==question.value]
        falseBranch = data[data[question.column]!=question.value]
        return trueBranch,falseBranch

    def predict(self,data,probability=False):
        if isinstance(data,pd.Series):
            data=data.to_frame().T
        result=[]
        for row in data.iterrows():
            row=row[1]
            node=self.rootNode
            while not node.leafNode:
                if row[node.question.column]==node.question.value:
                    node=node.tNode
                else:
                    node=node.fNode
            if probability:
                result.append(node.prediction)
            else:
                result.append(max(node.prediction, key=node.prediction.get,default=0))
        return r

from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
df = pd.read_csv("depression.csv")
columns=["Timestamp","Year","Label","Scale","Gender","Age","Residence","RelationshipStatus","FinancialState","Institute","FamilyRelation","Pressure","AcademicResult","LivingPlace","SupportedBy","SocialMedia","InferiorityComplex","MealSatisfaction","Health","PositiveActivity","Sleephour"]
df.columns=columns
df = df.sample(frac=1).reset_index(drop=True)
Mood = df["Label"]
df.drop(["Timestamp","Label","Scale"],axis=1,inplace=True)
columns=["Year","Gender","Residence","RelationshipStatus","FinancialState","Institute","FamilyRelation","Pressure","AcademicResult","LivingPlace","SupportedBy","SocialMedia","InferiorityComplex","MealSatisfaction","Health","PositiveActivity","Sleephour"]
df_Enc = pd.get_dummies(df,columns = columns)
x_train, x_test, y_train, y_test= train_test_split(df_Enc, Mood, test_size= 0.3, random_state=10)
pd.options.mode.chained_assignment = None 
DT=DecisionTree()
DT.fit(x_train,y_train)
y_pred = DT.predict(x_train)
accuracy_score(y_train, y_pred)*100
y_predicted = DT.predict(x_test)
accuracy_score(y_test, y_predicted)*100

from sklearn.tree import DecisionTreeClassifier
x_train, x_test, y_train, y_test= train_test_split(df_Enc, Mood, test_size= 0.3, random_state=10)
dTF = DecisionTreeClassifier(max_leaf_nodes=10,random_state=0)
dTF.fit(x_train,y_train)
y_pred = dTF.predict(x_train)
y_predicted = dTF.predict(x_test)
accuracy_score(y_train, y_pred)*100
accuracy_score(y_test, y_predicted)*100


