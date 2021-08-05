#%%
import numpy as np
import pandas as pd
from BaseModel import BaseModel
import seaborn as sns

def loadData():
    #Load Data  and Rename Columns
    df = pd.read_csv("data/concreteData.csv")
    df.columns = ["Cement","Slag","Ash","Water","Plasticizer","CoarseAgg","FineAgg","Age","CompressiveStrength"]
    return np.array(df.iloc[:,:-1]),np.array(df["CompressiveStrength"]),df
#Summary Stastics
    #visualize distrubution
    #visualize boxplots for distributions
def featureDistributions(df):
    sns.displot(df,x="Cement",bin)

X,y,df = loadData()
featureDistributions(df)

#Outlier Detection

def removeOutliers(df):
    X,y = df.iloc[:,:-1],df["CompressiveStrength"]
    X = np.log(X)
    return X.head()

    #take log of data in loadData, and eliminate outliers
    #use root, or box-cox
    #then standardize and pass to ML functions
#Feature Relationships, Coorelations with each other and Target


# %%
