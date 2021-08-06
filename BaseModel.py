from featureSelection import selectFeatures
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import linear_model
from sklearn.model_selection import train_test_split,GridSearchCV
from featureEngineering import *

class BaseModel:
    def __init__(self,transform=True):
        np.random.seed(13)
        if transform:
            df = selectFeatures()
            self.X,self.y = df.iloc[:,:-1],df["CompressiveStrength"]
        else:
            self.X,self.y,_ = self.loadData()
        self.X_train,self.X_validation,self.X_test,self.y_train,self.y_validation,self.y_test = self.split(self.X,self.y)
        #Standardize Data
        self.standardizationParams = {"mean":None,"std":None}
        self.X_train = self.standardize(self.X_train,useParams=False)
        self.X_validation = self.standardize(self.X_validation)
        self.X_test = self.standardize(self.X_test)
        #Build data for grid Search
        self.X_grid = np.concatenate((self.X_train,self.X_validation))
        self.y_grid = np.concatenate((self.y_train,self.y_validation))
        #Init Class Specific Variables
        self.modelType = None
        self.hyperParameterSearchSpace = None

    def loadData(self):
        #Load Data  and Rename Columns
        df = pd.read_csv("data/concreteData.csv")
        df.columns = ["Cement","Slag","Ash","Water","Plasticizer","CoarseAgg","FineAgg","Age","CompressiveStrength"]
        return np.array(df.iloc[:,:-1]),np.array(df["CompressiveStrength"]),df

    def standardize(self,X,useParams=True):
        if useParams:
            return (X-self.standardizationParams["mean"])/self.standardizationParams["std"]
        else:
            self.standardizationParams["mean"]=X.mean(axis=0)
            self.standardizationParams["std"]=X.std(axis=0)
            return (X-self.standardizationParams["mean"])/self.standardizationParams["std"]
  
    def split(self,X,y):
        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.30)
        X_validation,X_test,y_validation,y_test = train_test_split(X_test,y_test,test_size=0.5)
        return X_train,X_validation,X_test,y_train,y_validation,y_test
    
    def score(self,y,yHat):
        rSquared = metrics.r2_score(y,yHat)
        mae = metrics.mean_absolute_error(y,yHat)
        mae_PercentOfMean = mae/np.mean(y)
        return  [rSquared,mae,mae_PercentOfMean]

    def baselineMean(self,X,y):
        X_train,X_validation,X_test,y_train,y_validation,y_test = self.split(X,y)
        return self.score(y_validation,np.full_like(y_validation,np.mean(y_train)))
        
    def baselineLinearRegression(self,X,y):
        X_train,X_validation,X_test,y_train,y_validation,y_test = self.split(X,y)
        testModel = linear_model.LinearRegression()
        testModel.fit(X_train,y_train)
        yHat = testModel.predict(X_validation)
        return self.score(y_validation,yHat)
    
    def fit(self,X,y,hyperparameters=None):
        if hyperparameters:
            self.model = self.modelType(**hyperparameters)
            self.model.fit(X,y)
        else:
            self.model = self.modelType()
            self.model.fit(X,y)    

    def predict(self,X):
        return self.model.predict(X)

    def getHyperparameters(self,hyperparameters=None,updateModel=True):
        #Use random search to narrow down parameters to gridSearch
        if hyperparameters:
            gridSearcher = GridSearchCV(self.modelType(),hyperparameters)
            gridSearcher.fit(self.X_grid,self.y_grid)
        else:
            gridSearcher = GridSearchCV(self.modelType(),self.hyperParameterSearchSpace)
            gridSearcher.fit(self.X_grid,self.y_grid)

        if updateModel:
            self.model = gridSearcher.best_estimator_
        return gridSearcher.best_params_,self.score(gridSearcher.best_estimator_.predict(self.X_validation),self.y_validation)

    def estimatePerformance(self):
        return self.score(self.model.predict(self.X_test),self.y_test)
# %%
