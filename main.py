#%%
import numpy as np
import pandas as pd
from Models import models

def getScores(transform=False):
    output = []
    for model in models.keys():
        currentModel = models[model](transform)
        bestParams,validationMetrics = currentModel.getHyperparameters()
        output.append({
            "ModelType:":model,
            "ModelParams:":bestParams,
            "ModelValidationMetrics:":validationMetrics,
            "EstimatedPerformance:": currentModel.estimatePerformance()
        })
    return output

def displayScores(scores):
    for score in scores:
        print(f'{score["ModelType:"]}:Performance on Test Set')
        print(f'R-Squared:{round(score["EstimatedPerformance:"][0],2)} MeanAbsoluteError(MAE):{round(score["EstimatedPerformance:"][1],2)} MAE Adjusted(% Error):{int(round(score["EstimatedPerformance:"][2],2)*100)}')
        print("------------------------")

# %%
