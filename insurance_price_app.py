#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# ## ASSIGNMENT-MEDICAL INSURANCE COST PREDICTION
# ### Name-SHAMAL KADBE

# In[94]:


import warnings
warnings.filterwarnings('ignore')
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
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[22]:


df=pd.read_csv('MED_INS.csv')
df


# In[ ]:





# In[23]:


df.head()


# In[24]:


df.tail()


# In[ ]:





# In[25]:


df.info()


# In[ ]:





# In[26]:


df.describe().T


# In[ ]:





# In[27]:


df.isnull().sum()


# In[ ]:





# In[28]:


df.duplicated().sum()


# In[29]:


df.drop_duplicates(inplace=True)


# In[30]:


df.duplicated().sum()


# In[31]:


features = ['sex', 'smoker', 'region']

plt.subplots(figsize=(20, 10))
for i, col in enumerate(features):
    plt.subplot(1, 3, i + 1)

    x = df[col].value_counts()
    plt.pie(x.values,
            labels=x.index,
            autopct='%1.1f%%')

plt.show()


# In[ ]:





# In[32]:


features = ['sex', 'smoker', 'region']

plt.subplots(figsize=(20, 10))
for i, col in enumerate(features):
    plt.subplot(2, 2, i + 1)
    df.groupby(col)['charges'].mean().astype(float).plot.bar()
plt.show()


# In[ ]:





# In[33]:


import seaborn as sns

features = ['age', 'bmi']

plt.subplots(figsize=(17, 7))
for i, col in enumerate(features):
    plt.subplot(1, 2, i + 1)
    sns.scatterplot(data=df, x=col,
                   y='charges',
                   hue='smoker')
plt.show()


# In[ ]:





# In[34]:


df.drop_duplicates(inplace=True)

sns.boxplot(y=df['age'])  # specify x= or y= depending on orientation
plt.show() 


# In[35]:


sns.boxplot(y=df['bmi'])  # specify x= or y= depending on orientation
plt.show() 


# ####  Due to the presence of outliers present in bmi column we need to treat the outliers by replacing the values with mean
# #### as the bmi column consists of continuous data.

# In[36]:


Q1=df['bmi'].quantile(0.25)
Q2=df['bmi'].quantile(0.5)
Q3=df['bmi'].quantile(0.75)
iqr=Q3-Q1
lowlim=Q1-1.5*iqr
upplim=Q3+1.5*iqr
print(lowlim)
print(upplim)


# In[37]:


from feature_engine.outliers import ArbitraryOutlierCapper
arb=ArbitraryOutlierCapper(min_capping_dict={'bmi':13.6749},max_capping_dict={'bmi':47.315})
df[['bmi']]=arb.fit_transform(df[['bmi']])
sns.boxplot(y=df['bmi'])
plt.show()


# In[38]:


df['bmi'].skew()


# In[39]:


df['age'].skew()


# In[42]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_scaled = scaler.fit_transform(df)


# In[41]:


df['sex'] = df['sex'].map({'male': 0, 'female': 1}).fillna(-1)
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
df['region'] = df['region'].map({'northwest': 0, 'northeast': 1, 'southeast': 2, 'southwest': 3})
df[['sex', 'smoker', 'region']] = df[['sex', 'smoker', 'region']].astype(int)
df


# ### Model Building

# In[45]:


train_df, validation_df = train_test_split(df,test_size=0.2,random_state=2)
linreg = LinearRegression()
linreg.fit(raw_x_train, raw_y_train)


# In[44]:


raw_x_train = df.drop(['charges'], axis = 1)
raw_y_train = df['charges']

raw_x_val = validation_df.drop(['charges'], axis = 1)
raw_y_val = validation_df['charges']


# In[ ]:





# #### Linear Regression Model

# In[46]:


# Linear Regression model
linear_model_raw = LinearRegression()
linear_model_raw.fit(raw_x_train, raw_y_train)
raw_y_pred_train_lr = linear_model_raw.predict(raw_x_train)
raw_y_pred_val_lr = linear_model_raw.predict(raw_x_val)


# In[47]:


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


# #### Decision Tree Model

# In[48]:


DecisionTree_raw = DecisionTreeRegressor()
DecisionTree_raw.fit(raw_x_train, raw_y_train)
raw_y_pred_train_lr = DecisionTree_raw.predict(raw_x_train)
raw_y_pred_val_lr = DecisionTree_raw.predict(raw_x_val)


# In[49]:


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


# #### Random Forest Model

# In[50]:


RandomForest_raw = RandomForestRegressor()
RandomForest_raw.fit(raw_x_train, raw_y_train)
raw_y_pred_train_lr = RandomForest_raw.predict(raw_x_train)
raw_y_pred_val_lr = RandomForest_raw.predict(raw_x_val)


# In[51]:


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


# #### Gradient Boosting Model

# In[52]:


GradientBoosting_raw = GradientBoostingRegressor()
GradientBoosting_raw.fit(raw_x_train, raw_y_train)
raw_y_pred_train_lr = GradientBoosting_raw.predict(raw_x_train)
raw_y_pred_val_lr = GradientBoosting_raw.predict(raw_x_val)


# In[53]:


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


# In[54]:


SVR_raw = SVR()
SVR_raw.fit(raw_x_train, raw_y_train)
raw_y_pred_train_lr = SVR_raw.predict(raw_x_train)
raw_y_pred_val_lr = SVR_raw.predict(raw_x_val)


# In[55]:


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


# In[56]:


KNN_raw = KNeighborsRegressor()
KNN_raw.fit(raw_x_train, raw_y_train)
raw_y_pred_train_lr = KNN_raw.predict(raw_x_train)
raw_y_pred_val_lr = KNN_raw.predict(raw_x_val)


# In[57]:


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


# In[ ]:


from xgboost import XGBRegressor

xgb_raw = XGBRegressor()
xgb_raw.fit(raw_x_train, raw_y_train)

raw_y_pred_train_lr = xgb_raw.predict(raw_x_train)
raw_y_pred_val_lr = xgb_raw.predict(raw_x_val)


# In[58]:


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


# In[ ]:





# In[ ]:


Model	                     Train RMSE	     Test RMSE	    Train R²	   Test R²	    Overfitting (Y/N)
Linear Regression			36593573.827	41887124.441      0.750          0.722	        NO
Decision Tree				195464.714	    829351.564        0.998          0.994          NO
Random Forest			 	3412833.019     4025855.537       0.976          0.970          NO
Gradient Boosting			14527368.295    16850084.375	  0.900          0.888          NO
SVR                         161719207.873   165283635.654    -0.103         -0.094          NO
KNeighbors Regressor        85322798.638    89222848.477      0.417          0.409          NO
XGB Regressor               1089606.102     1801570.288       0.992          0.988          NO



# In[59]:


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


# In[60]:


import joblib
joblib.dump(best_dtree, 'best_decision_tree_model.pkl')


# In[ ]:





# In[ ]:





# In[63]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("model.pkl")  # Make sure model.pkl is in the same directory

st.title("Medical Insurance Cost Predictor")

st.markdown("### Enter Patient Details")

# Example input fields - customize based on your model
age = st.number_input("Age", min_value=1, max_value=100, value=30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.slider("Number of Children", 0, 5, 1)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Prepare input DataFrame
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'bmi': [bmi],
    'children': [children],
    'smoker': [smoker],
    'region': [region]
})

# Preprocessing if needed (e.g., one-hot encoding) must match training phase

# Predict button
if st.button("Predict Insurance Cost"):
    try:
        prediction = model.predict(input_data)
        st.success(f"Predicted Insurance Cost: ₹ {prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Error: {e}")


# In[91]:


import joblib

# Save
joblib.dump(model, "model.pkl")

# Load in Streamlit app
model = joblib.load("model.pkl")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




