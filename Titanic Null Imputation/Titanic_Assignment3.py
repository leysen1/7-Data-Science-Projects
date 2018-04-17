#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 17:19:22 2018

@author: charlotteleysen
"""

'''
Extended description: There are two columns with missing data in the Titanic dataset: 
    Age and Cabin. Your first task is to reason what is the meaning of the missing data 
    for these two columns: did the creators of the dataset leave missing values on purpose 
    or not? Secondly, if you think there is any column whose missing data was not left there 
    on purpose and that we could benefit from having, think of some strategy to impute it 
    (https://www.wikiwand.com/en/Imputation_(statistics)). The most trivial strategy is to 
    impute the missing values with some global aggregation: the mean, the median, or the mode. 
    Can you do better?
    
Questions:
    Why is Age and Cabin data missing? Is it on purpose?
    How could we impute the missing data?

Grading (5 points): Notice that this dataset comes from a Kaggle competition. The most natural 
way to evaluate such a problem would be to just impute the data and rank the final result. However, 
the purpose of this assignment is not to have the highest Kaggle score, but to reason about the data. 
For the first question, there is a "correct" answer, so 2 points will be given (1 for each column). 
For the second question, 1 point will be given if at least one smart strategy is included, and another 
1 point if there is a code implementation for it. The last point is a subjective global measure that 
can be scored by structuring the notebook in a way that is pleasant to read, checking that the code 
follows PEP8, and have innovative or interesting ideas that don't appear on the first page of Google 
results.

Extra advice: I recommend using the dataset from Kaggle, since it is more complete.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fancyimpute import KNN

#Import Data
filepath = "/Users/charlotteleysen/Google Drive/*PROJECTS/IE/Term 2/Python/Assignments/Titanic Assignment 3/"
df = pd.read_csv(filepath+"train.csv")

df.columns
len(df)

Questions:
Why is Age and Cabin data missing? Is it on purpose?

df.isnull().sum(axis = 0)

From the data, it appears as though the Embarked column also contains missing values, but only 2 out of 891 observations.
On the other hand, Age has 177 (20%) missing, and Cabin has 687 (77%) missing values.
I will do some analysis of the data to figure out why this may be:
  
df.dropna().describe()

Cabin:   
Percentage of NAs by Pclass and Sex:
round(df.loc[df.Cabin.isnull(),:].groupby(["Pclass",'Sex']).size() / df.groupby(["Pclass",'Sex']).size() * 100)
round(df.loc[df.Cabin.isnull(),:].groupby(["Pclass"]).size() / df.groupby(["Pclass"]).size() * 100)
Percentage of NAs by Parch:
round(df.loc[df.Cabin.isnull(),:].groupby(["Parch"]).size() / df.groupby(["Parch"]).size() * 100)

Most Cabin nulls are related to passengers from 2nd and 3rd class. Males Cabin are on average more missing that females,
while 2nd class passengers are missing 91% of their Cabins, and 3rd class passengers as missing 98%. In addition, 
passengers of large families (5-6) do not have any recorded Cabins.

In conclusion, the Cabin nulls are probably due to the fact that lower class passenger weren't assigned particular Cabins. 
There may be just a lower class section of the ship which these passengers got sent to and were assigned rooms when they 
got there. Meanwhile, first class passengers would have rooms assigned when they bought their tickets. These nulls would 
not have been created on purpose.

Age:    
plt.hist(df.Age.dropna(), bins = 20)
By Pclass and Sex:
round(df.loc[df.Age.isnull(),:].groupby(["Pclass",'Sex']).size() / df.groupby(["Pclass",'Sex']).size() * 100)
By Parch:
round(df.loc[df.Age.isnull(),:].groupby(["Parch"]).size() / df.groupby(["Parch"]).size() * 100)
plt.hist(df.Parch[df.Age.isnull()])
By Survived:
round(df.loc[df.Age.isnull(),:].groupby(["Survived"]).size() / df.groupby(["Survived"]).size() * 100)
By Fare:
df_Age_null = df.loc[df.Age.isnull(),:]
df_Age_null.groupby(pd.cut(df_Age_null['Fare'], np.linspace(0, 600, 7))).agg({'PassengerId':'count'})
plt.hist(df.Fare[df.Age.isnull()], bins = 40)

Generally again more third class passengers had null age values, although 1st class came in second place, while 2nd class
passengers had fewest age values missing. There is not much distinguishing Age nulls in the Parch or Survived variables 
either. Finally, most of the NAs in age are from passengers who paid less than 100 for their ticket.

In conclusion, there seems to be no clear reason from the data for so many Ages to be missing. For this reason, it may be 
to conclude that this data was left out on purpose by Kaggle to may the competition more difficult. Especially as age
is an important demographic variable. However, as most ages are missing from passengers that paid less than 100 for their 
tickets, it implies that poorer people are missing their age. During the time of the titanic, poorer people may not 
have had good, precise records of their birth dates.

How could we impute the missing data?

Instead of using the mean, median or mode to impute the NAs, we could use KNN imputation or regression. Here I have 
chosen to use KNN. For the non numeric variables, I first had to convert them to integers. I chose to order the Cabin
Letters in alphabetical order.

#KNN Imputation - Age, Cabin, Embarked

    #Create new columns for KNN
df['Cabin_Letter'] = df['Cabin'].astype(str).str[0]
df["Cabin_Letter"] = df["Cabin_Letter"].replace(['A','B','C','D','E','F','G','T'], [1,2,3,4,5,6,7,8])
df.loc[df.Cabin_Letter == 'n','Cabin_Letter'] = None
df["Embarked_New"] = df["Embarked"].replace(['C', 'S', 'Q'], [1,2,3])

num_vars = ['Age', 'Cabin_Letter', 'Embarked_New','Survived', 'Pclass','SibSp','Parch',  'Fare']
df_impute = pd.DataFrame(KNN(k=3).complete(df.loc[:,num_vars]))
df_impute.columns = num_vars
    
    #Round the predictions
df_impute.Age = df_impute.Age.round()    
df_impute.Cabin_Letter = df_impute.Cabin_Letter.round()
df_impute.Embarked_New = df_impute.Embarked_New.round()
    
    # Update df
df_impute.isnull().sum(axis = 0)
df = pd.concat([df.drop(num_vars, axis = 1),df_impute], axis = 1)

    #Update original Cabin and Embarked columns again 
df["Cabin"] = df["Cabin_Letter"].replace([1,2,3,4,5,6,7,8],['A','B','C','D','E','F','G','T'])
df["Embarked"] = df["Embarked_New"].replace([1,2,3],['C', 'S', 'Q'])
df.drop(["Embarked_New","Cabin_Letter"], inplace = True, axis = 1)

    #Check
df.isnull().sum(axis = 0)
