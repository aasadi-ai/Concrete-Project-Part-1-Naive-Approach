# __Regression Project: Concrete Data__
## __Table of Contents:__
* Project Overview and Methodology
* File Descriptions
* Usage

## __Project Overview and Methodology__:
We attempt to predict the compressive strength of concrete from its ingredients and age. This project uses the dataset found [here](https://www.kaggle.com/maajdl/yeh-concret-data).
As [maajdl](https://www.kaggle.com/maajdl) indicates:
> Concrete is the most important material in civil engineering.
> Concrete compressive strength is a highly nonlinear function of age and ingredients.<br/>

We have good reason to believe that the problem is important and non-trivial.
Our approach is explained in detail in the ML_Walkthrough file but can be summed up as follows:
* Make sure the features are normally distributed
* Remove outliers
* Generate features by taking ratios and products of existing features
* Trim features using Recursive Feature Elimination
* Train several regression models and choose best model

## __File Descriptions__
1. Data Folder: Contains Concrete dataset
2. BaseModel: Implements all functions needed to train and evlauate an ML model
3. Models: A class containing all regression models which inherit from BaseModel
4. exploratoryDataAnalysis: Implements functions needed to describe and vizualize data
5. featureEngineering: Transforms features, removes outliers and creates new features
6. featureSelection: Contains functions used to trim features and vizualize relationships between features
7. main: Trains the models and displays their performance
8. __ML_Walkthrough: An interactive walkthrough of our ML project__

## __Usage__
Follow along with __ML_Walkthrough__ or call main.py to run the models and review their performance.
