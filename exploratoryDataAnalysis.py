#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def loadData():
    #Load Data  and Rename Columns
    df = pd.read_csv("data/concreteData.csv")
    df.columns = ["Cement","Slag","Ash","Water","Plasticizer","CoarseAgg","FineAgg","Age","CompressiveStrength"]
    return np.array(df.iloc[:,:-1]),np.array(df["CompressiveStrength"]),df

#Summary Stastics
def summaryStatistics(df):
    return df.describe(),df.dtypes
    
def featureDistributions(df):
    for columnName in df.columns.tolist():
        sns.displot(df,x=columnName)

def boxPlots(df):
    for columnName in df.columns.tolist():
        plt.figure()
        sns.boxplot(data=df,x=columnName)
        plt.show()
# %%
