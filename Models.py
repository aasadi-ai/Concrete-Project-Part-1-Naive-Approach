#%%
import numpy as np
import pandas as pd
from BaseModel import BaseModel
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

class SVM(BaseModel):
    def __init__(self,transform=True):
        super().__init__(transform)
        self.modelType = SVR
        self.model = None
        self.hyperParameterSearchSpace = {
            "kernel": ["linear","rbf"],
            "degree":[3],
            "C":[0.01,1]
        }

class Linear(BaseModel):
     def __init__(self,transform=True):
        super().__init__(transform)
        self.modelType = Ridge
        self.model = None
        self.hyperParameterSearchSpace = {"alpha":[i/10 for i in range(0,10,1)]}

class DecisionTree(BaseModel):
    def __init__(self,transform=True):
        super().__init__(transform)
        self.modelType = DecisionTreeRegressor
        self.model = None
        self.hyperParameterSearchSpace = {
            "max_depth":[i for i in range(1,16)],
            "min_samples_leaf":[i for i in range(2,10)]
        } 

class KNN(BaseModel):
    def __init__(self,transform=True):
        super().__init__(transform)
        self.modelType = KNeighborsRegressor
        self.model = None
        self.hyperParameterSearchSpace = {"n_neighbors":[i for i in range(1,11)]}     

models = {"SVM":SVM,"KNN":KNN,"DecisionTree":DecisionTree,"LinearRidge":Linear}
# %%
