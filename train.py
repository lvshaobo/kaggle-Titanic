# -*- coding: utf-8 -*-
'''
Created on 2016年5月31日

@author: lvshaobo
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from xlwt.Style import rotation_func
from numpy import dtype



data_train = pd.read_csv("/home/lvshaobo/git/kaggle/Titanic/data/train.csv")

# section 1
"""
fig = plt.figure()
print data_train.Survived.value_counts()
print type(data_train.Survived.value_counts())
data_train.Survived.value_counts().plot(kind='bar')

#plt.plot(data_train.Survived.value_counts(), 'r', label='Survived')
plt.title("Survived")
plt.ylabel("numbers")


fig = plt.figure()
plt.scatter(data_train.Survived, data_train.Age, marker='s', color='r')
plt.title("Age-Survived")
plt.ylabel("Age")
plt.grid(b='on', which='major', axis='y', color='r')

fig = plt.figure()
data_train.Age[data_train.Pclass == 1].plot(kind='kde', label='Pclass 1')
data_train.Age[data_train.Pclass == 2].plot(kind='kde', label='Pclass 2')
data_train.Age[data_train.Pclass == 3].plot(kind='kde', label='Pclass 3')
plt.legend()


fig = plt.figure()
data_train.Embarked.value_counts().plot(kind='bar')

plt.show()
"""
#section 2
"""
Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df = pd.DataFrame({u'Survived': Survived_1, u'UnSurvived': Survived_0})
df.plot(kind='bar', stacked=False, color='rb')
plt.xlabel('Pclass')

Survived_0 = data_train.Sex[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Sex[data_train.Survived == 1].value_counts()
df = pd.DataFrame({u'Survived': Survived_1, u'UnSurvived': Survived_0})
df.plot(kind='bar', stacked=True, color='rb')
plt.xlabel('Sex')

Survived_1 = data_train.Survived[data_train.Pclass == 1].value_counts()
Survived_2 = data_train.Survived[data_train.Pclass == 2].value_counts()
Survived_3 = data_train.Survived[data_train.Pclass == 3].value_counts()
df = pd.DataFrame({u'Pclass=1': Survived_1, u'Pclass=2': Survived_2, u'Pclass=3': Survived_3})
df.plot(kind='bar', stacked=True, color='rgb')
plt.xlabel('Survived')
"""
Survived_1 = data_train.Survived[data_train.Sex == 'female'].value_counts()
print type(Survived_1)
print Survived_1
#print Survived_1
Survived_2 = data_train.Survived[data_train.Sex == 'male'].value_counts()
df = pd.DataFrame({u'female': Survived_1, u'male': Survived_2})
df.plot(kind='bar', stacked=True, color='bm')
plt.xlabel('Survived')
plt.xticks(range(2), ("UnSurvived", "Survived"), rotation=0)
plt.show()

# section 3
# print data_train[['Age']]
# print type(data_train[['Age', 'SibSp']])
# print type(data_train[['Age', 'SibSp']]['Age'])
#print data_train['Cabin']
def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df
data_train = set_Cabin_type(data_train)

print type(data_train)
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')
#print dummies_Cabin
df = pd.concat([dummies_Cabin])
print type(df)
print df
print df.as_matrix()
print type(df.as_matrix())