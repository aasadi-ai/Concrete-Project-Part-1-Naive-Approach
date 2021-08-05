#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from exploratoryDataAnalysis import loadData,featureDistributions,boxPlots

#Outlier Detection
def transformFeatures(df):
    X = df.iloc[:,:-1]
    X = np.sqrt(X)
    df.iloc[:,:-1] = X
    return df

def findAndRemoveOutliers(df,zThreshhold=2.5):
    X = df.iloc[:,:-1]
    X = (X-X.mean(axis=0))/X.std(axis=0)
    mask = (np.abs(X) <= zThreshhold).all(1)
    return df[mask].iloc[:,:-1],df[mask]["CompressiveStrength"],df[mask]

X,y,df = loadData()
# %%
