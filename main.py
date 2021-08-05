#%%
from Models import models

def getScores():
    output = []
    for model in models.keys():
        currentModel = models[model]()
        bestParams,validationMetrics = currentModel.getHyperparameters()
        output.append({
            "ModelType:":model,
            "ModelParams:":bestParams,
            "ModelValidationMetrics:":validationMetrics,
            "EstimatedPerformance:": currentModel.estimatePerformance()
        })
    return output

getScores()
#%%