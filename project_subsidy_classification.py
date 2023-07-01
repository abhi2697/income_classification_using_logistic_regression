# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 13:47:57 2023

@author: abppa
"""
"""subsidy Inc . delivers subsides to individuals based on their income
Accurate income data is one of the hardest piece of data to obtain across the world
subsidy Inc .has obtained a large data set of authenicated data on individual income,demographic parameters ,and a few financial parameters.
Subsidy Inc wishes us to :
    develop an income classifier system for individuals
    the objectives is to 
    simplify the data system by reducing the number of variable to be studied ,without 
    sacrificing too much of accuracy.such a system woukd help subsidy Inc. in planning
    subsidy outlay,monitoring and preventing misuse
"""
#importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
#import data
data_income=pd.read_csv('income.csv')
#Creating a copy of original data
data=data_income.copy()

#Exploratory data analysis:
    #1.Getting to know the data
    #2.Data preprocessing (Missing values)
    #3.Cross tables and data visualization
#Getting to know the data
#to check variables data type
print(data.info())
# check for missing values
data.isnull()
print('Data columns with null values :\n',data.isnull().sum())
#summary of numerical variable
summary_num=data.describe()
print(summary_num)
#summary of categorical variables
summary_categorical=data.describe(include='O')
print(summary_categorical)
#frequency of each categories
data['JobType'].value_counts()
data['occupation'].value_counts()
data['EdType'].value_counts()
data['maritalstatus'].value_counts()
data['relationship'].value_counts()

#checking for unique classes
print(np.unique(data['JobType']))
print(np.unique(data['occupation']))
# there  exits ' ?' instead of nan
"""
go back and read the data by including "na_values [' ?']" to consider ' ?' as nan """
data=pd.read_csv('income.csv',na_values=[" ?"])

### data pre processig
data.isnull().sum()

missing =data[data.isnull().any(axis=1)]
#axis=1 => to consider at least one column value is missing
""" point to note :
    1.missing values in jobtype= 1809
    2.missing value in occupation =1816
    3.there was 1809 rows where two specific
    columns i.e. occupation & JobType have missing values
    4.1816-1809 =7= you still have occupation unfilled for 7 rows.
    because,jobtype is never worked"""
data2=data.dropna(axis =0)
data3=data2.copy()

summary_num2=data2.describe()
# realtionship between independent variables
correlation = data2.corr()
#none of the values lie near to one there is no corelation 
data2.columns
gender= pd.crosstab(index=data['gender'] ,columns='count',normalize=True)
print(gender)
#gender vs salary status
gender_salstat=pd.crosstab(index= data2["gender"], columns=data2['SalStat'],margins=True,normalize='index') 
print(gender_salstat)

#============================================================
#frequency distribution of 'salary status'
salstat=sns.countplot(data2['SalStat'])
""" 75% of people's salary status is <=50,000
&25% of people's salary status is >50000"""
##histogram of age
sns.displot(data2['age'],bins=10,kde=False)
#people with age 20-45 ages are high in frequency
###### box plot -Age vs Salary status########
sns.boxplot('SalStat','age',data=data2)
data2.groupby('SalStat')['age'].median()
##people with 35-50 age are more likely to earn >5000q
##Exploratory data analysis (tbc)

    
"""
age_salstat=pd.crosstab(index= data2["age"], columns=data2['SalStat'],margins=True,normalize='index')
print(age_salstat)
JobType_salstat=pd.crosstab(index= data2["JobType"], columns=data2['SalStat'],margins=True,normalize='index')
print(JobType_salstat)
age_salstat=pd.crosstab(index= data2["age"], columns=data2['SalStat'],margins=True,normalize='index')
print(age_salstat)
age_salstat=pd.crosstab(index= data2["age"], columns=data2['SalStat'],margins=True,normalize='index')
print(age_salstat)
age_salstat=pd.crosstab(index= data2["age"], columns=data2['SalStat'],margins=True,normalize='index')
print(age_salstat)"""
    
#LogisticRegression()
#Reindexing the salary status to 0,1
tmp=data2['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})


new_data=pd.get_dummies(data2,drop_first=True)

#storing the column names
columns_list=list(new_data.columns)
print(columns_list)

#separating the input names from data
feature=list(set(columns_list)-set(['SalStat']))

#storing the output values in y
y=new_data['SalStat'].values
print(y)

#storing the input 
x=new_data[feature].values
print(x)
#splitting the data into train and test
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
#make an instance of the Model
logistic=LogisticRegression(C=1.0,n_jobs=1,max_iter=10000)

#fitting the values for x and y
logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_

#prediction from test data
prediction=logistic.predict(test_x)
print(prediction)

# confusion matrix
cf_matrix=confusion_matrix(test_y,prediction)
print(cf_matrix) 

#Calculating the accuracy
ac_score=accuracy_score(test_y,prediction)
print(ac_score)

#printing the misclassified values from prediction
print('Misclassified samples :%d'% (test_y !=prediction).sum())
 #logistic regression - removig insingnificant variables
#reindexing the salary status names as 0,1
data3['SalStat']=data3['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data3['SalStat'])

cols=['gender','nativecountry','race','JobType']
new_data=data3.drop(cols,axis=1)

new_data=pd.get_dummies(new_data,drop_first=True)
#staring the column names
columns_list=list(new_data.columns)
print(columns_list)

#separting the input names from data
features=list(set(columns_list)-set(['SalStat']))
print(features)

#storing the output values in y
y=new_data['SalStat'].values
print(y)
#storing the input from input features
x=new_data[features].values
print(x)
#splitting the data into train and test
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.25,random_state=0)

#make an instance of the model
logistic1=LogisticRegression(n_jobs=1,max_iter=10000)

#fitting the values for x and y
logistic1.fit(train_x,train_y)
#prediction from test data
prediction1=logistic1.predict(test_x)
print(prediction1)
print(test_y)
#Calculating the accuracy
cf_matrix=confusion_matrix(test_y,prediction1)
print(cf_matrix) 
ac_mat=accuracy_score(test_y,prediction1)
print(ac_mat)
#printing the misclassified values from prediction
#
#KNN
#

#importing the library of KNN
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

KNN_classifier=KNeighborsClassifier(n_neighbors=5)
#fitting the values
KNN_classifier.fit(train_x,train_y)
#prediction the test values with model
prediction2=KNN_classifier.predict(test_x)
#performance metric check
cf_matrix2=confusion_matrix(test_y,prediction2)
print("Original values","\n",cf_matrix2)

#calculating the acuracy
accuracy_score2=accuracy_score(test_y,prediction2)
print(accuracy_score2)

print('Misclassified sample: %d' %(test_y !=prediction2).sum())
Misclassified_sample=[]
for i in range(1,20):
    
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x,train_y)
    pred_i=knn.predict(test_x)
    Misclassified_sample.append((test_y !=pred_i).sum())
    
print(Misclassified_sample)
