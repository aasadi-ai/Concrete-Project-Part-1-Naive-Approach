#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import feature_selection
from featureEngineering import *
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeRegressor

def visualizeCorrelations(df,targetOnly=True):
    correlations = df.corr()
    if not targetOnly:
        sns.heatmap(correlations)
        return correlations
    correlationsWithTarget = correlations["CompressiveStrength"].sort_values(ascending=False)
    sns.boxplot(correlationsWithTarget)
    return correlationsWithTarget

def selectFeatures():
    df = applyFeatureEngineering()
    X,y = df.iloc[:,:-1],df["CompressiveStrength"]
    featureSelector = RFE(estimator=DecisionTreeRegressor(),n_features_to_select=12)
    featureSelector.fit(X,y)
    bestFeatures = X.columns[featureSelector.support_].tolist()
    bestFeatures.append("CompressiveStrength")
    return df[bestFeatures]



# %%
