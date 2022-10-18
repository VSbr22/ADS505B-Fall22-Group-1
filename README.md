# Maximizing Profit in E-Commerce Sales 

This is a final project for ADS505: https://github.com/VSbr22/ADS505B-Fall22-Group-1

#### -- Project Status: [Completed]

## Project Intro
The purpose of this project is to present the process used to find and predict profitable products sold from a Global Superstore. Various models were used for regression and classification to determine if products could be predicted to be profitable or not. These models include KNN,SVR, Logistic Regression, . The Global Superstore data is collected from 01/01/2011 through 12/31/2014 providing years of repetitive buying and purchasing behavbiors. In order to better serve company stakeholders the project was established to determine how to increase profit by being able to provide recommendations based on model performance. 

### Team 3
* George Garcia
* Summer Purschke
* Vannesa Salazar


### Methods Used
* Data preprocessing
* Exploratory Data Analysis
* Data Visualization
* Predictive Modeling


### Technologies
* Python
* Pandas, jupyternotebook


## Project Description
- Dataset provides many insights on the buying and purchasing behaviors of the consumers of the global superstore within a span of approximately four years. 
- Key stakeholders would be very interested in how to increase profits to the gloabl store, while also reducing profits. 
- By predicting the most profitable products recommendations can be made as to what items to keep in stock and what items the global superstore should no longer carry. 
- By establishing the baselines of a product and being able to predict its outcome of profitablity will be greatly beneficial to the company longterm. 
- Dataset provided by Kaggle 
-(https://www.kaggle.com/datasets/apoorvaappz/global-super-store-dataset)
-The data is taken from global online orders in the time frame beginning 01/01/2011 through 12/31/2014.


## Needs of this project

- data exploration/descriptive statistics
- data preocessing/data cleaning
- statistical modeling
- techincal report/notebook
- presentation- Pitch deck
- voice recordings

# Required Python Packages
*Basics
import pandas as pd
import numpy as np
import seaborn as sns
import scipy

*Visualization
import matplotlib.pylab as plt
%matplotlib inline

*Modeling
import statsmodels.formula.api as sm
import statsmodels.tools.tools as stattools
from scipy.stats import skew

from dmba import regressionSummary, exhaustive_search
from dmba import backward_elimination, forward_selection, stepwise_selection
from dmba import adjusted_r2_score, AIC_score, BIC_score
from dmba import plotDecisionTree, gainsChart, liftChart
from dmba import classificationSummary, regressionSummary

from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, BayesianRidge, SGDRegressor
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz, plot_tree
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix,r2_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn import metrics

from wsgiref.simple_server import WSGIRequestHandler

* Set basic options for consistent output
PRECISION = 2
np.set_printoptions(precision = PRECISION)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.precision', PRECISION)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')

* Set Matplotlib defaults for consistent visualization look 'n' feel
FONTSIZE_S = 10
FONTSIZE_M = 12
FONTSIZE_L = 14
plt.style.use('default')
plt.rcParams['figure.titlesize'] = FONTSIZE_L
plt.rcParams['figure.figsize'] = (9, 9 / (16 / 9))
plt.rcParams['figure.subplot.left'] = '0.1'
plt.rcParams['figure.subplot.bottom'] = '0.1'
plt.rcParams['figure.subplot.top'] = '0.9'
plt.rcParams['figure.subplot.wspace'] = '0.4'
plt.rcParams['lines.linewidth'] = '2'
plt.rcParams['axes.linewidth'] = '2'
plt.rcParams['axes.titlesize'] = '8'
#plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = FONTSIZE_M
plt.rcParams['xtick.labelsize'] = FONTSIZE_S
plt.rcParams['ytick.labelsize'] = FONTSIZE_S
plt.rcParams['grid.linewidth'] = '1'
plt.rcParams['legend.fontsize'] = FONTSIZE_S
plt.rcParams['legend.title_fontsize'] = FONTSIZE_S

## Getting Started

1. Clone this repository using the raw data.
2. Add additional code or make changes and push new code into repo. Code should be ran prior to adding to the repository in order to ensure cohesiveness. 


## Featured Notebooks/Analysis/Deliverables
* [Presentation slides ](https://docs.google.com/presentation/d/1hQ1v_VHhQWNttZtbZUAn5VgFi_z1K_R1qDin0TLAbKQ/edit#slide=id.gc6f9e470d_0_0)


## Data Pre-Processing
* Format was changed for features to remove extra white space and special characters.
* Normalization was preformed on the numeric variables, to counter skewed variables. 
* Profit was changed to binary 0 and 1
* Ordinal encoding was created for shipping variables. 
* Categorical data was changed under dytpes to reflect as category type. 




## Features Selection

* Correlation to Target 



## Modeling & Evaluation
* Logistic Regression
* KNearestNeighbors
* Support Vector Machine
