import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import xgboost as XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pandas.plotting import scatter_matrix
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score , mean_absolute_error as mae
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

sns.set()


df=pd.read_excel('C:\DataScience\Medical Insurance cost prediction.xlsm')
df.head()

df.tail()
df.info()
df.describe().T
df.isnull().sum()
df.duplicated().sum()
df.drop_duplicates(inplace=True)
features = ['sex', 'smoker', 'region']

plt.subplots(figsize=(20, 10))
for i, col in enumerate(features):
    plt.subplot(1, 3, i + 1)

    x = df[col].value_counts()
    plt.pie(x.values,
            labels=x.index,
            autopct='%1.1f%%')

plt.show()
features = ['sex', 'smoker', 'region']

plt.subplots(figsize=(20, 10))
for i, col in enumerate(features):
    plt.subplot(2, 2, i + 1)
    df.groupby(col)['charges'].mean().astype(float).plot.bar()
plt.show()


import seaborn as sns

features = ['age', 'bmi']

plt.subplots(figsize=(17, 7))
for i, col in enumerate(features):
    plt.subplot(1, 2, i + 1)
    sns.scatterplot(data=df, x=col,
                   y='charges',
                   hue='smoker')
plt.show()

df.drop_duplicates(inplace=True)

sns.boxplot(y=df['age'])  # specify x= or y= depending on orientation
plt.show() 

sns.boxplot(y=df['bmi'])  # specify x= or y= depending on orientation
plt.show() 

Q1=df['bmi'].quantile(0.25)
Q2=df['bmi'].quantile(0.5)
Q3=df['bmi'].quantile(0.75)
iqr=Q3-Q1
lowlim=Q1-1.5*iqr
upplim=Q3+1.5*iqr
print(lowlim)
print(upplim)


df['bmi'].skew()
df['age'].skew()
scaler = StandardScaler()
df['sex'] = df['sex'].map({'male': 0, 'female': 1}).fillna(-1)
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
df['region'] = df['region'].map({'northwest': 0, 'northeast': 1, 'southeast': 2, 'southwest': 3})
df[['sex', 'smoker', 'region']] = df[['sex', 'smoker', 'region']].astype(int)
df


train_df, validation_df = train_test_split(df,test_size=0.2,random_state=2)
linreg = LinearRegression()



raw_x_train = train_df.drop(['charges'], axis = 1)
raw_y_train = train_df['charges']

raw_x_val = validation_df.drop(['charges'], axis = 1)
raw_y_val = validation_df['charges']


# Linear Regression model
linear_model_raw = LinearRegression()
linear_model_raw.fit(raw_x_train, raw_y_train)
raw_y_pred_train_lr = linear_model_raw.predict(raw_x_train)
raw_y_pred_val_lr = linear_model_raw.predict(raw_x_val)

print("Accuracy Scores for Linear Regression model on raw data")
raw_val_rmse = mean_squared_error(raw_y_val, raw_y_pred_val_lr)

raw_train_rmse = mean_squared_error(raw_y_train, raw_y_pred_train_lr)

raw_train_lr_r2s = r2_score(raw_y_train, raw_y_pred_train_lr)

raw_val_lr_r2s = r2_score(raw_y_val, raw_y_pred_val_lr)

raw_val_lr_r2s = r2_score(raw_y_val, raw_y_pred_val_lr)

raw_lr_mae = mean_absolute_error(raw_y_val, raw_y_pred_val_lr)

mse = mean_squared_error(raw_y_val, raw_y_pred_val_lr)

n = raw_x_val.shape[0]  # number of observations
p = raw_x_val.shape[1]  # number of features
adjusted_r2 = 1 - (1 - raw_train_lr_r2s) * (n - 1) / (n - p - 1)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", raw_lr_mae)
print("Root Mean Squared Error(Test) :", raw_val_rmse)
print("Root Mean Squared Error(Train) :", raw_train_rmse)
print("Adjusted R² Score:", adjusted_r2)
print("R-squared Score (Train) :", raw_train_lr_r2s)
print("R-squared Score (Test) :", raw_val_lr_r2s)
print("**************" * 7)



DecisionTree_raw = DecisionTreeRegressor()
DecisionTree_raw.fit(raw_x_train, raw_y_train)
raw_y_pred_train_lr = DecisionTree_raw.predict(raw_x_train)
raw_y_pred_val_lr = DecisionTree_raw.predict(raw_x_val)
print("Accuracy Scores for Decision Tree model on raw data")
raw_val_rmse = mean_squared_error(raw_y_val, raw_y_pred_val_lr)

raw_train_rmse = mean_squared_error(raw_y_train, raw_y_pred_train_lr)

raw_train_lr_r2s = r2_score(raw_y_train, raw_y_pred_train_lr)

raw_val_lr_r2s = r2_score(raw_y_val, raw_y_pred_val_lr)

raw_val_lr_r2s = r2_score(raw_y_val, raw_y_pred_val_lr)

raw_lr_mae = mean_absolute_error(raw_y_val, raw_y_pred_val_lr)

mse = mean_squared_error(raw_y_val, raw_y_pred_val_lr)

n = raw_x_val.shape[0]  # number of observations
p = raw_x_val.shape[1]  # number of features
adjusted_r2 = 1 - (1 - raw_train_lr_r2s) * (n - 1) / (n - p - 1)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", raw_lr_mae)
print("Root Mean Squared Error(Test) :", raw_val_rmse)
print("Root Mean Squared Error(Train) :", raw_train_rmse)
print("Adjusted R² Score:", adjusted_r2)
print("R-squared Score (Train) :", raw_train_lr_r2s)
print("R-squared Score (Test) :", raw_val_lr_r2s)
print("**************" * 7)


RandomForest_raw = RandomForestRegressor()
RandomForest_raw.fit(raw_x_train, raw_y_train)
raw_y_pred_train_lr = RandomForest_raw.predict(raw_x_train)
raw_y_pred_val_lr = RandomForest_raw.predict(raw_x_val)

print("Accuracy Scores for Random Forest model on raw data")
raw_val_rmse = mean_squared_error(raw_y_val, raw_y_pred_val_lr)

raw_train_rmse = mean_squared_error(raw_y_train, raw_y_pred_train_lr)

raw_train_lr_r2s = r2_score(raw_y_train, raw_y_pred_train_lr)

raw_val_lr_r2s = r2_score(raw_y_val, raw_y_pred_val_lr)

raw_val_lr_r2s = r2_score(raw_y_val, raw_y_pred_val_lr)

raw_lr_mae = mean_absolute_error(raw_y_val, raw_y_pred_val_lr)

mse = mean_squared_error(raw_y_val, raw_y_pred_val_lr)

n = raw_x_val.shape[0]  # number of observations
p = raw_x_val.shape[1]  # number of features
adjusted_r2 = 1 - (1 - raw_train_lr_r2s) * (n - 1) / (n - p - 1)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", raw_lr_mae)
print("Root Mean Squared Error(Test) :", raw_val_rmse)
print("Root Mean Squared Error(Train) :", raw_train_rmse)
print("Adjusted R² Score:", adjusted_r2)
print("R-squared Score (Train) :", raw_train_lr_r2s)
print("R-squared Score (Test) :", raw_val_lr_r2s)
print("**************" * 7)



GradientBoosting_raw = GradientBoostingRegressor()
GradientBoosting_raw.fit(raw_x_train, raw_y_train)
raw_y_pred_train_lr = GradientBoosting_raw.predict(raw_x_train)
raw_y_pred_val_lr = GradientBoosting_raw.predict(raw_x_val)


print("Accuracy Scores for Gradient Boosting model on raw data")

raw_val_rmse = mean_squared_error(raw_y_val, raw_y_pred_val_lr)

raw_train_rmse = mean_squared_error(raw_y_train, raw_y_pred_train_lr)

raw_train_lr_r2s = r2_score(raw_y_train, raw_y_pred_train_lr)

raw_val_lr_r2s = r2_score(raw_y_val, raw_y_pred_val_lr)

raw_val_lr_r2s = r2_score(raw_y_val, raw_y_pred_val_lr)

raw_lr_mae = mean_absolute_error(raw_y_val, raw_y_pred_val_lr)

mse = mean_squared_error(raw_y_val, raw_y_pred_val_lr)

n = raw_x_val.shape[0]  # number of observations
p = raw_x_val.shape[1]  # number of features
adjusted_r2 = 1 - (1 - raw_train_lr_r2s) * (n - 1) / (n - p - 1)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", raw_lr_mae)
print("Root Mean Squared Error(Test) :", raw_val_rmse)
print("Root Mean Squared Error(Train) :", raw_train_rmse)
print("Adjusted R² Score:", adjusted_r2)
print("R-squared Score (Train) :", raw_train_lr_r2s)
print("R-squared Score (Test) :", raw_val_lr_r2s)
print("**************" * 7)


SVR_raw = SVR()
SVR_raw.fit(raw_x_train, raw_y_train)
raw_y_pred_train_lr = SVR_raw.predict(raw_x_train)
raw_y_pred_val_lr = SVR_raw.predict(raw_x_val)


print("Accuracy Scores for SVR model on raw data")

raw_val_rmse = mean_squared_error(raw_y_val, raw_y_pred_val_lr)

raw_train_rmse = mean_squared_error(raw_y_train, raw_y_pred_train_lr)

raw_train_lr_r2s = r2_score(raw_y_train, raw_y_pred_train_lr)

raw_val_lr_r2s = r2_score(raw_y_val, raw_y_pred_val_lr)

raw_val_lr_r2s = r2_score(raw_y_val, raw_y_pred_val_lr)

raw_lr_mae = mean_absolute_error(raw_y_val, raw_y_pred_val_lr)

mse = mean_squared_error(raw_y_val, raw_y_pred_val_lr)

n = raw_x_val.shape[0]  # number of observations
p = raw_x_val.shape[1]  # number of features
adjusted_r2 = 1 - (1 - raw_train_lr_r2s) * (n - 1) / (n - p - 1)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", raw_lr_mae)
print("Root Mean Squared Error(Test) :", raw_val_rmse)
print("Root Mean Squared Error(Train) :", raw_train_rmse)
print("Adjusted R² Score:", adjusted_r2)
print("R-squared Score (Train) :", raw_train_lr_r2s)
print("R-squared Score (Test) :", raw_val_lr_r2s)
print("**************" * 7)


KNN_raw = KNeighborsRegressor()
KNN_raw.fit(raw_x_train, raw_y_train)
raw_y_pred_train_lr = KNN_raw.predict(raw_x_train)
raw_y_pred_val_lr = KNN_raw.predict(raw_x_val)


print("Accuracy Scores for KNeighbors Regressor model on raw data")

raw_val_rmse = mean_squared_error(raw_y_val, raw_y_pred_val_lr)

raw_train_rmse = mean_squared_error(raw_y_train, raw_y_pred_train_lr)

raw_train_lr_r2s = r2_score(raw_y_train, raw_y_pred_train_lr)

raw_val_lr_r2s = r2_score(raw_y_val, raw_y_pred_val_lr)

raw_val_lr_r2s = r2_score(raw_y_val, raw_y_pred_val_lr)

raw_lr_mae = mean_absolute_error(raw_y_val, raw_y_pred_val_lr)

mse = mean_squared_error(raw_y_val, raw_y_pred_val_lr)

n = raw_x_val.shape[0]  # number of observations
p = raw_x_val.shape[1]  # number of features
adjusted_r2 = 1 - (1 - raw_train_lr_r2s) * (n - 1) / (n - p - 1)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", raw_lr_mae)
print("Root Mean Squared Error(Test) :", raw_val_rmse)
print("Root Mean Squared Error(Train) :", raw_train_rmse)
print("Adjusted R² Score:", adjusted_r2)
print("R-squared Score (Train) :", raw_train_lr_r2s)
print("R-squared Score (Test) :", raw_val_lr_r2s)
print("**************" * 7)


from xgboost import XGBRegressor

xgb_raw = XGBRegressor()
xgb_raw.fit(raw_x_train, raw_y_train)

raw_y_pred_train_lr = xgb_raw.predict(raw_x_train)
raw_y_pred_val_lr = xgb_raw.predict(raw_x_val)



print("Accuracy Scores for XGB Regressor model on raw data")

raw_val_rmse = mean_squared_error(raw_y_val, raw_y_pred_val_lr)

raw_train_rmse = mean_squared_error(raw_y_train, raw_y_pred_train_lr)

raw_train_lr_r2s = r2_score(raw_y_train, raw_y_pred_train_lr)

raw_val_lr_r2s = r2_score(raw_y_val, raw_y_pred_val_lr)

raw_val_lr_r2s = r2_score(raw_y_val, raw_y_pred_val_lr)

raw_lr_mae = mean_absolute_error(raw_y_val, raw_y_pred_val_lr)

mse = mean_squared_error(raw_y_val, raw_y_pred_val_lr)

n = raw_x_val.shape[0]  # number of observations
p = raw_x_val.shape[1]  # number of features
adjusted_r2 = 1 - (1 - raw_train_lr_r2s) * (n - 1) / (n - p - 1)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", raw_lr_mae)
print("Root Mean Squared Error(Test) :", raw_val_rmse)
print("Root Mean Squared Error(Train) :", raw_train_rmse)
print("Adjusted R² Score:", adjusted_r2)
print("R-squared Score (Train) :", raw_train_lr_r2s)
print("R-squared Score (Test) :", raw_val_lr_r2s)
print("**************" * 7)

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# Step 1: Define parameter grid
param_grid = {
    'max_depth': [3, 5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['squared_error', 'friedman_mse']  # For regression
}

# Step 2: Initialize base model
dtree = DecisionTreeRegressor(random_state=42)

# Step 3: Grid search
grid_search = GridSearchCV(estimator=dtree,
                           param_grid=param_grid,
                           cv=5,
                           scoring='neg_mean_squared_error',
                           n_jobs=-1)

# Step 4: Fit on scaled training data
grid_search.fit(raw_x_train, raw_y_train)

# Step 5: Get best model
best_dtree = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

# Step 6: Evaluate
y_pred = best_dtree.predict(raw_x_val)
mse = mean_squared_error(raw_y_val, raw_y_pred_val_lr)
print("Test MSE with best Decision Tree:", mse)

import joblib
joblib.dump(best_dtree, 'best_decision_tree_model.pkl')

import streamlit as st
import joblib

import streamlit as st
import joblib
import os

# Load the model
model = joblib.load('best_decision_tree_model.pkl')

# Title
st.title("Medical Insurance Cost Predictor")

# Input fields
age = st.number_input("Age", min_value=0)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=0.0)
children = st.number_input("Number of Children", min_value=0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Convert inputs to model format
sex = 0 if sex == 'male' else 1
smoker = 1 if smoker == 'yes' else 0
region_dict = {'northwest': 0, 'northeast': 1, 'southeast': 2, 'southwest': 3}
region = region_dict[region]

# Predict button
if st.button("Predict Insurance Cost"):
    features = np.array([[age, sex, bmi, children, smoker, region]])
    prediction = DecisionTree_raw.predict(features)
    st.success(f"Predicted Insurance Cost: ₹{prediction[0]:,.2f}")

import os
final_model='best_decision_tree_model.pkl'
filename='scaler.pkl'
if os.path.exists(filename):
    os.remove(filename)
joblib.dump(final_model,filename)

