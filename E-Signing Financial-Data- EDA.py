# -*- coding: utf-8 -*-
"""
Title: Predicting the Likelihood of E-Signing a Loan Based on Financial
       History - EDA
Created on Tue Jul 28 16:38:38 2020
@author: Jayasooryan TM
"""

# importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the data
dataset = pd.read_csv(r'P39-Financial-Data.csv')

# Inspecting data for missing values and outliers
print(dataset.head())
print(dataset.describe())
print(dataset.isnull().sum())

# Plotting histogram for features to get insights

df = dataset.drop(columns=['entry_id', 'pay_schedule','e_signed'])

fig = plt.figure(figsize=(15,12))

plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(df.shape[1]):
    plt.subplot(6, 3, i+1)
    f = plt.gca()
    f.set_title(df.columns.values[i])
    
    vals = np.size(df.iloc[:, i].unique())
    if vals >= 100:
        vals = 100
    
    plt.hist(df.iloc[:,i], bins=vals, color='#3F5D7D')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Plotting the correlation bar chart between features and target

df.corrwith(dataset.e_signed).plot.bar(figsize=(20,10),
                                       title='Correlation with e_signed',
                                       fontsize=20, rot=45, grid=True,
                                       color=['red', 'green', 'blue', 'orange',
                                              'yellow'])

# Plotting the correlation matrix between features and target

sns.set(style="white")

corr = df.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, axs = plt.subplots(figsize = (18,15))

cmap=sns.diverging_palette(220, 10, as_cmap = True)

sns.heatmap(corr, mask = mask, cmap = cmap, vmax =0.3, center = 0,
            square = True, linewidth = 0.5, cbar_kws = {"shrink":0.5})

