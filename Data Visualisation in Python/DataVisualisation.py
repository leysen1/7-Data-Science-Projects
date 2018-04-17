#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 16:20:21 2018

@author: charlotteleysen
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
sns.set_palette("Set2", 10)

f_path = '/Users/charlotteleysen/Google Drive/*PROJECTS/IE/Term 2/Python/Assignments/Visualation Census Data/data/'
df = pd.read_csv(f_path + 'adult.data.txt', header=None)

df.describe()
df.info()

# CLEAN THE DATA WITH PROPER NULLS AND COLUMN NAMES

col_names = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education_num',
    'marital_status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital_gain',
    'capital_loss',
    'hours_per_week',
    'native_country',
    'income']

df.columns = col_names
df.head()

df.isnull()
df.apply(lambda x: sum(x.isnull()))
# Only the last row has nulls in half the columns


365*7.2*24
object_cols = list(df.select_dtypes(['object']).columns)

# Remove white space
df[object_cols] = df[object_cols].apply(lambda x: x.str.strip())

df_weather["date"] = 

# Change all ? to None values
for i in object_cols:
    df.loc[df.loc[:, i] == '?', i] = None
    df.loc[df.loc[:, i].isnull(), i] = None

# Change df objects to categories
df[object_cols] = df[object_cols].apply(lambda x: x.astype('category'))
df.info()

# Check out factor levels
factor_levels = df[object_cols].apply(lambda x: list(x.unique()))
factor_levels[5]

df.groupby('workclass').count()

# Check the null in each column again
df.apply(lambda x: sum(x.isnull()))

# VISUALISE NUMERICAL CORRELATION
numeric_cols = list(df.select_dtypes(
    ['float']).columns) + list(df.select_dtypes(['int']).columns)
corr = df[numeric_cols].corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
sns.pairplot(corr)

# PLOT VISUALATION
# The relationships between Age, Work Hours and earning
# <= or > 50k, displaying the effect of Sex, Race and Education as well


# Scatter Plots of data
sns.set(font_scale=1.4)

sns.set_palette("viridis", 4)
a = sns.FacetGrid(df, col="income", hue="race", size=5, margin_titles=True)
a.map(
    sns.regplot,
    "hours_per_week",
    "age",
    scatter_kws={
        's': 40,
        'alpha': 0.3,
        'edgecolor': "white",
        'lw': .7},
    data=df,
    fit_reg=False)
a.add_legend()

sns.set_palette("viridis", 1)
a = sns.FacetGrid(df, col="income", hue="sex", size=5, margin_titles=True)
a.map(
    sns.regplot,
    "hours_per_week",
    "age",
    scatter_kws={
        's': 40,
        'alpha': 0.3,
        'edgecolor': "white",
        'lw': .7,
    },
    data=df,
    fit_reg=False)
a.add_legend()

sns.set_palette("viridis", 20)
a = sns.FacetGrid(
    df,
    col="income",
    hue="education",
    size=5,
    margin_titles=True)
a.map(
    sns.regplot,
    "hours_per_week",
    "age",
    scatter_kws={
        's': 40,
        'alpha': 0.5,
        'edgecolor': "white",
        'lw': .7,
    },
    data=df,
    fit_reg=False,
    label='big')
a.add_legend()


from bokeh.plotting import figure, show, output_file
from bokeh.models.widgets import Panel
from bokeh.models.widgets import Tabs
from bokeh.models import CategoricalColorMapper, ColumnDataSource

TOOLS = "hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,reset,tap,box_select,lasso_select"

# Make a CategoricalColorMapper object: color_mapper
source = ColumnDataSource(
    df.loc[:, ['hours_per_week', 'age', 'sex', 'race', 'education']])

color_mapper1 = CategoricalColorMapper(factors=['Female', 'Male'],
                                       palette=['red', 'blue'])
p.circle('weight', 'mpg', source=source,
         color=dict(field='origin', transform=color_mapper),
         legend='origin')
p1 = figure()
p1.scatter(
    'hours_per_week',
    'age',
    source=source,
    color=dict(
        field='sex',
        transform=color_mapper),
    fill_alpha=0.6,
    legend='sex')

show(p1)


df.race.unique()

color_mapper2 = CategoricalColorMapper(
    factors=[
        'White',
        'Black',
        'Asian-Pac-Islander',
        'Amer-Indian-Eskimo',
        'Other'],
    palette=[
        'white',
        'black',
        'red',
        'yellow',
        'green'])

p2 = figure()
p2.scatter(
    'hours_per_week',
    'age',
    source=source,
    color=dict(
        field='race',
        transform=color_mapper2),
    fill_alpha=0.6,
    line_color='white',
    legend='race')
show(p2)


a = list(df.education.unique())
for value in a:
    print(a)
color_mapper2 = CategoricalColorMapper(
    factors=[
        'Bachelors',
        'HS-grad',
        '11th',
        'Masters',
        '9th',
        'Some-college',
        'Assoc-acdm',
        'Assoc-voc',
        '7th-8th',
        'Doctorate',
        'Prof-school',
        '5th-6th',
        '10th',
        '1st-4th',
        'Preschool',
        '12th'],
    palette=[
        'firebrick',
        'sienna',
        'gold',
        'green',
        'mediumspringgreen',
        'deepskyblue',
        'royalblue',
        'plum',
        'm',
        'slategray',
        'moccasin',
        'silver',
        'navy',
        'darkcyan'
        'mediumpurple',
        'black'])
colormap = {
    'Bachelors': 'firebrick',
    'HS-grad': 'sienna',
    '11th': 'gold',
    'Masters': 'green',
    '9th': 'mediumspringgreen',
    'Some-college': 'deepskyblue',
    'Assoc-acdm': 'royalblue',
    'Assoc-voc': 'plum',
    '7th-8th': 'm',
    'Doctorate': 'slategray',
    'Prof-school': 'moccasin',
    '5th-6th': 'silver',
    '10th': 'darkcyan',
    '1st-4th': 'navy',
    'Preschool': 'mediumpurple',
    '12th': 'black'}
name = str(df.education.unique())
from bokeh.palettes import Spectral16
colors = [colormap[x] for x in df['education']]

p3 = figure(tools=TOOLS)
p3.circle(df.hours_per_week, df.age, color=colors,
          fill_alpha=0.6,
          line_color='white', legend=name)
show(p3)
factor_levels['workclass'].keys()
# Panels and view plots as tabs


tab1 = Panel(child=p1, title='sex')
tab2 = Panel(child=p2, title='race')

layout = Tabs(tabs=[tab2])
show(layout)
