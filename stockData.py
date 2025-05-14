import bentoml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import sklearn.linear_model as skl
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

from mlxtend.feature_selection import SequentialFeatureSelector
import gc

originalData = pd.read_csv('SP_SPX, 30.csv')
rangeInDayList = []

cleanData = originalData.dropna(axis = 1)
cleanData = cleanData.drop('time', axis = 1)
# cleanData = originalData

#pop = originalData[['open', 'high', 'low', 'SMA #1', 'VWAP']]
targetColumn = cleanData['SMA #1'] # SMA 1 will always be the 10 SMA
cleanData = cleanData.drop('SMA #1', axis = 1)

print(cleanData.keys())

sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
varianceSelections = sel.fit_transform(cleanData)

X = cleanData
y = targetColumn

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, y, test_size = .20, random_state = 100)
forestRegressor = RandomForestRegressor(n_jobs = -1)

forestRegressor.fit(X_Train, Y_Train)
forestRegressor.score(X_Test, Y_Test)
forwardFeatureSelection = SequentialFeatureSelector(forestRegressor,
                                                    k_features = 2,
                                                    forward = True,
                                                    floating= True, 
                                                    verbose = 2,
                                                    cv = 5
                                                    ).fit(X_Train, Y_Train)

forwardFeatureSelection.k_feature_idx_

featureSelection = []

for idx in forwardFeatureSelection.k_feature_idx_:
    featureSelection.append(cleanData.keys()[idx])
    
forwardFeatureSelection.k_feature_names_


X = cleanData[featureSelection]

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, y, train_size = 0.8, test_size = .20, random_state = 100)

X_Train, X_Valid, Y_Train, Y_Valid = train_test_split(X, y, train_size = 0.9, test_size = .10, random_state = 100)

X_Train.shape, Y_Train.shape
X_Test.shape, Y_Test.shape

model = linear_model.LinearRegression()
model.fit(X_Train, Y_Train)

Y_pred = model.predict(X_Test)
print('Coefficients: ', model.coef_)
print('Intercept: ', model.intercept_)
print('Mean squared error (MSE): %.2f' % r2_score(Y_Test, Y_pred))

print('Coefficients of determination (R^2):  %.2f' % r2_score(Y_Test, Y_pred))

model.predict(X_Test)
saved_model = bentoml.sklearn.save_model("SMA_10_prediction", model)
print(f'Model Saved: ', {saved_model})

#tag="sma_10_prediction:ltemf3o2q2vomqfq