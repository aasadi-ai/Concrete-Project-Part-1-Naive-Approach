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

def featureCreation(df):
    outputDF = df
    columns = df.columns.tolist()[:-1]
    for column0 in columns:
        print(column0)
        outputDF[f"{column0}_Sqrd"] = np.square(df[column0])
        for column1 in columns:
            if column0!=column1:
                outputDF[f"{column0}/{column1}"] = df[column0]/(df[column1]+0.1)
                outputDF[f"{column0}*{column1}"] = df[column0]*df[column1]
    return outputDF

# X,y,df = loadData()
# df = transformFeatures(df)
# _,_,df = findAndRemoveOutliers(df)
# df = featureCreation(df)
# df.head()
# %%
