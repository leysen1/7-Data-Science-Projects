#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 21:23:08 2018

@author: charlotteleysen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fancyimpute import KNN

# Import Data
filepath = "/Users/charlotteleysen/Google Drive/*PROJECTS/IE/Term 2/Python/Assignments/Titanic Assignment 3/"
df = pd.read_csv(filepath + "train.csv")

# Questions:
# Why is Age and Cabin data missing? Is it on purpose?

df.isnull().sum(axis=0)

# From the data, it appears as though the Embarked column also contains
# missing values, but only 2 out of 891 observations. On the other hand,
# Age has 177 (20 % ) missing, and Cabin has 687 (77%) missing values. I
# will do some analysis of the data to figure out why this may be:

df.dropna().describe()

# Cabin:
# Percentage of NAs by Pclass and Sex:
round(df.loc[df.Cabin.isnull(), :].groupby(
    ["Pclass"]).size() / df.groupby(["Pclass"]).size() * 100)
# Percentage of NAs by Parch:
round(df.loc[df.Cabin.isnull(), :].groupby(
    ["Parch"]).size() / df.groupby(["Parch"]).size() * 100)


# How could we impute the missing data?

# Instead of using the mean, median or mode to impute the NAs, we could
# use KNN imputation or regression. Here I have chosen to use KNN.

# Create new columns for KNN
num_vars = [
    'Age',
    'Survived',
    'Pclass',
    'SibSp',
    'Parch',
    'Fare']
df_impute = pd.DataFrame(KNN(k=3).complete(df.loc[:, num_vars]))
df_impute.columns = num_vars

# Round the predictions
df_impute.Age = df_impute.Age.round()

# Update df
df_impute.isnull().sum(axis=0)
df = pd.concat([df.drop(num_vars, axis=1), df_impute], axis=1)

