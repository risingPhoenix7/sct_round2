# %% [markdown]
# # Importing needed libraries

# %%
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from hyperopt import hp, fmin, tpe, Trials
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.datasets import make_regression
#IMPORT MODULES
import pandas as pd
import numpy as np
import time
import pickle
from hyperopt import fmin, tpe, Trials
# SENTIMENT ANALYSIS USING VADER
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# %%
%pip install --upgrade setuptools

# %%
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

# %%
%pip install pkg_resources

# %% [markdown]
# # Loading dataset

# %%
df = pd.read_csv('harddrive/harddrive.csv')
print(df.shape)
df.head()

# %%
df = df.loc[:, ~df.isnull().all()]


# number of hdd
print("number of hdd:", df['serial_number'].value_counts().shape) 

# number of different types of harddrives
print("number of different harddrives", df['model'].value_counts().shape)

failed_hdds = df.loc[df.failure==1]["serial_number"]

print("Number of failed hdd: ", len(failed_hdds))

df = df.loc[df["serial_number"].isin(failed_hdds)]



# %% [markdown]
# # Finding the RUL (Remaining Useful Life)

# %%
df["end_date"] = df.groupby("serial_number")["date"].transform("max")

df["end_date"] = pd.to_datetime(df["end_date"])
df["date"] = pd.to_datetime(df["date"])


df["date_diff"] = df["end_date"] - df["date"]
df["date_diff"].describe()

# %% [markdown]
# # Check number of null entries

# %%
df.isnull().sum()

# %% [markdown]
# # Getting rid of NaN values

# %%
df_notna = df[df.columns[~(df.isna().sum().values/len(df) > 0.05)]]
df_notna.dropna(inplace=True)
df = df_notna.copy()
df.reset_index(inplace=True, drop=True)

# %% [markdown]
# # Check result

# %%
df.isnull().sum()

# %%
df = df.rename(columns={
    "smart_5_normalized": "REAllOCATED_SECTOR_COUNT_N",
    "smart_187_normalized": "REPORTED_UNCORRECTABLE_ERRORS_N",
    "smart_188_normalized": "COMMAND_TIMEOUT_N",
    "smart_197_normalized": "CURRENT_PENDING_SECTOR_COUNT_N",
    "smart_198_normalized": "OFFLINE_UNCORRECTABLE_N",
    "smart_9_normalized": "POWER_ON_HOURS_N",
    "smart_5_raw": "REAllOCATED_SECTOR_COUNT_R",
    "smart_187_raw": "REPORTED_UNCORRECTABLE_ERRORS_R",
    "smart_188_raw": "COMMAND_TIMEOUT_R",
    "smart_197_raw": "CURRENT_PENDING_SECTOR_COUNT_R",
    "smart_198_raw": "OFFLINE_UNCORRECTABLE_R",
    "smart_9_raw": "POWER_ON_HOURS_R",
    "date": "DATE",
    "serial_number": "SERIAL_NUMBER",
    "model": "MODEL",
    "capacity_bytes": "CAPACITY_BYTES",
    "failure": "FAILURE",
    "smart_10_normalized": "Spin_Retry_Count_N",
    "smart_10_raw": "Spin_Retry_Count_R",
    "smart_12_normalized": "Power_Cycle_Count_N",
    "smart_12_raw": "Power_Cycle_Count_R",
    "smart 192_normalized": "Power-Off_Retract_Count_N",
    "smart_192_raw": "Power-Off_Retract_Count_R",
    "smart_240_raw": "Head_Flying_Hours_R",
    "smart_3_normalized": "Spin_Up_Time_N",
    "smart_3_raw": "Spin_Up_Time_R",
    "smart_4_normalized": "Start_Stop_Count_N",
    "smart_4_raw": "Start_Stop_Count_R",
    "smart_7_normalized": "Seek_Error_Rate_N",
    "smart_7_raw": "Seek_Error_Rate_R",
    "smart_241_normalized": "Total_LBAs_Written_N",
    "smart_242_normalized": "Total_LBAs_Read_N",
    "smart_199_normalized": "UDMA_CRC_Error_Count_N",
    "smart_199_raw": "UDMA_CRC_Error_Count_R",
    "smart_9_normalized": "Power_On",
    "smart_9_raw": "Power_On_Hours_R",
    "smart_194_normalized": "Temperature_Celsius_N",
    "smart_194_raw": "Temperature_Celsius_R",
    "smart_241_raw": "Total_LBAs_Written_R",
    "smart_242_raw": "Total_LBAs_Read_R",
    "smart_192_normalized": "Power-Off_Retract_Count_N",
    "smart_193_normalized": "Load_Cycle_Count_N",
    "smart_193_raw": "Load_Cycle_Count_R"

})


# %% [markdown]
# # Feature Engineering to create new features

# %%
df['POWER_ON_HOURS_UTILIZATION'] = df['Power_On_Hours_R'] / df['Power_On']
df['SPIN_RETRY_TO_POWER_CYCLE_RATIO'] = df['Spin_Retry_Count_R'] / df['Power_Cycle_Count_R']
df['REAllOCATED_SECTOR_ERROR_RATE'] = df['REAllOCATED_SECTOR_COUNT_R'] / df['REAllOCATED_SECTOR_COUNT_N']
df['CURRENT_PENDING_SECTOR_ERROR_RATE'] = df['CURRENT_PENDING_SECTOR_COUNT_R'] / df['CURRENT_PENDING_SECTOR_COUNT_N']
df['TEMPERATURE_DIFFERENCE'] = df['Temperature_Celsius_N'] - df['Temperature_Celsius_R']
# Assuming you have a column 'failure_date' representing the date of failure
# df['TIME_SINCE_LAST_FAILURE'] = (df['DATE'] - df['failure_date']).dt.days
df['MOVING_AVERAGE_READ_ERROR_RATE'] = df['REAllOCATED_SECTOR_ERROR_RATE'].rolling(window=30, min_periods=1).mean()
# Assuming you have a column 'manufacture_date' representing the date of manufacture
# df['DRIVE_AGE'] = (pd.Timestamp.now() - df['manufacture_date']).dt.days
df['REAllOCATED_SECTOR_ERROR_TREND'] = np.where(df['REAllOCATED_SECTOR_COUNT_R'].diff() > 0, 1, -1)
# Assuming you have a column 'failure' indicating whether the drive failed (1) or not (0)
failure_rates = df.groupby('SERIAL_NUMBER')['FAILURE'].mean()
df['FAILURE_PROBABILITY'] = df['SERIAL_NUMBER'].map(failure_rates)
# df['AVERAGE_USAGE_PER_DAY'] = df['Power_On_Hours_R'] / df['DRIVE_AGE']
# Assuming you have a column 'maintenance_date' representing the date of maintenance
# df['DAYS_SINCE_LAST_MAINTENANCE'] = (pd.Timestamp.now() - df['maintenance_date']).dt.days
# mean_performance = df.drop(['SERIAL_NUMBER', 'DATE'], axis=1).mean()
# df['RElATIVE_PERFORMANCE'] = df.drop(['SERIAL_NUMBER', 'DATE'], axis=1).mean(axis=1)


# %% [markdown]
# # Dropping unwanted columns, Splitting Data and Training Model

# %%
df = df.drop(['DATE', 'SERIAL_NUMBER', 'MODEL','end_date'], axis=1)
df.head()

# %%
Y = df["date_diff"].dt.days
X = df.drop(["date_diff"],axis=1)

# %%
from sklearn.model_selection import train_test_split

# Assuming you have your features in X and target variable in y
# Replace this with your actual data

# Perform train-test split (adjust test_size and random_state as needed)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Print the shapes of the resulting splits
print("Shapes - X_train:", X_train.shape, "X_test:", X_test.shape, "y_train:", y_train.shape, "y_test:", y_test.shape)


# %%
classifiers = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "Support Vector Machine": SVR(),
    "K-Nearest Neighbors": KNeighborsRegressor(),
    "Neural Network": MLPRegressor(),
    "Gaussian Process": GaussianProcessRegressor(),
}

# %% [markdown]
# # CHECK IMPORTANCE OF FEATURES

# %%

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X, Y)
feature_importances = model.feature_importances_

map_ = {}

for i, a in enumerate(X):
    map_[a] = feature_importances[i]
    
sorted(map_.items(), key = lambda x: x[1], reverse=True)

# %% [markdown]
# # TESTING A NUMBER OF REGRESSORS TO CHOOSE THE BEST MODEL

# %%
# Create an empty DataFrame to store the results
columns = ['Model', 'Run Time (minutes)', 'MAE', 'MSE', 'RMSE', 'R2']
df_models = pd.DataFrame(columns=columns)

# Loop through your regression models
for key, clf in classifiers.items():
    # STARTING TIME
    start_time = time.time()
    # TRAIN CLASSIFIER ON TRAINING DATA
    clf.fit(X_train, y_train)
    
    #SAVE THE TRAINED MODEL
    classifiers[key] = clf
    
    # MAKE PREDICTIONS USING CURRENT CLASSIFIER
    predictions = clf.predict(X_test)
    
    # CALCULATE REGRESSION METRICS
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)  # Calculate RMSE
    r2 = r2_score(y_test, predictions)

    row = {'Model': key,
           'Run Time (minutes)': round((time.time() - start_time) / 60, 2),
           'MAE': mae,
           'MSE': mse,
           'RMSE': rmse,
           'R2': r2
           }

    df_models = pd.concat([df_models, pd.DataFrame([row])], ignore_index=True)

# Sort the DataFrame by R-squared (R2) in descending order
df_models = df_models.sort_values(by='R2', ascending=False)

# PRINT THE MODELS WITH REGRESSION METRICS [SORTED]
print(df_models)

# %% [markdown]
# # RESULT - RANDOM FOREST PERFORMS BEST.
# 
# ## Further, we test Random forest with and without hyperparameter tuning

# %% [markdown]
# # Random Forest Without Hyper Parameter Tuning

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Creating Random Forest Regressor
RF_model = RandomForestRegressor(random_state=1)
RF_model.fit(X_train, y_train)

# Predicting on test data
y_pred_RF = RF_model.predict(X_test)

# Calculating Mean Squared Error
mse_RF = mean_squared_error(y_test, y_pred_RF)
print("Random Forest MSE:", mse_RF)

# Calculating R^2 score
score_RF = RF_model.score(X_test, y_test)
print("Random Forest R^2 score:", score_RF)

# Visualizing Actual vs Predicted values with different colors
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_RF, alpha=0.5, label='Predicted', color='blue')
plt.scatter(y_test, y_test, alpha=0.5, label='Actual', color='red')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values (Random Forest)')
plt.legend()
plt.show()

# %% [markdown]
# # Decision Tree Without HyperParameter Tuning

# %%
from sklearn.tree import DecisionTreeRegressor

# Creating Decision Tree Regressor
DT_model = DecisionTreeRegressor(random_state=1)
DT_model.fit(X_train, y_train)

# Predicting on test data
y_pred_DT = DT_model.predict(X_test)

# Calculating Mean Squared Error
mse_DT = mean_squared_error(y_test, y_pred_DT)
print("Decision Tree MSE:", mse_DT)

# Calculating R^2 score
score_DT = DT_model.score(X_test, y_test)
print("Decision Tree R^2 score:", score_DT)

# Visualizing Actual vs Predicted values with different colors
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_DT, alpha=0.5, label='Predicted', color='green')
plt.scatter(y_test, y_test, alpha=0.5, label='Actual', color='red')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values (Decision Tree)')
plt.legend()
plt.show()


# %% [markdown]
# # Random Forest Parameter Tuning With Hyperopt

# %%
seed=2
def objective(params):
    est=int(params['n_estimators'])
    md=int(params['max_depth'])
    msl=int(params['min_samples_leaf'])
    mss=int(params['min_samples_split'])
    model=RandomForestRegressor(n_estimators=est,max_depth=md,min_samples_leaf=msl,min_samples_split=mss)
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    score=mean_squared_error(y_test,pred)
    return score

def optimize(trial):
    params={'n_estimators':hp.uniform('n_estimators',10,100),
           'max_depth':hp.uniform('max_depth',5,20),
           'min_samples_leaf':hp.uniform('min_samples_leaf',1,5),
           'min_samples_split':hp.uniform('min_samples_split',2,6)}
    best=fmin(fn=objective,space=params,algo=tpe.suggest,trials=trial,max_evals=150,rstate=np.random.default_rng(seed=2))
    return best

trial=Trials()
best=optimize(trial)

# %%
print(best)

for t in trial.trials[:2]:
    print (t)

# %%
TID=[t['tid'] for t in trial.trials]
Loss=[t['result']['loss'] for t in trial.trials]
maxd=[t['misc']['vals']['max_depth'][0] for t in trial.trials]
nest=[t['misc']['vals']['n_estimators'][0] for t in trial.trials]
min_ss=[t['misc']['vals']['min_samples_split'][0] for t in trial.trials]
min_sl=[t['misc']['vals']['min_samples_leaf'][0] for t in trial.trials]

hyperopt_rfr=pd.DataFrame({'tid':TID,'loss':Loss,
                          'max_depth':maxd,'n_estimators':nest,
                          'min_samples_split':min_ss, 'min_samples_leaf':min_sl})

# %%
plt.subplots(3,2,figsize=(10,10))
plt.subplot(3,2,1)
sns.scatterplot(x='tid',y='max_depth',data=hyperopt_rfr)
plt.subplot(3,2,2)
sns.scatterplot(x='tid',y='loss',data=hyperopt_rfr)
plt.subplot(3,2,3)
sns.scatterplot(x='tid',y='n_estimators',data=hyperopt_rfr)
plt.subplot(3,2,4)
sns.scatterplot(x='tid',y='min_samples_leaf',data=hyperopt_rfr)
plt.subplot(3,2,5)
sns.scatterplot(x='tid',y='min_samples_split',data=hyperopt_rfr)

plt.tight_layout()

# %%
best_params = {
    'n_estimators': int(best['n_estimators']),
    'max_depth': int(best['max_depth']),
    'min_samples_leaf': int(best['min_samples_leaf']),
    'min_samples_split': int(best['min_samples_split']),
    'random_state': 2  # Assuming you want to set the random state
}

rfr_opt = RandomForestRegressor(**best_params)
rfr_opt.fit(X_train,y_train)
pred_rfr_opt=rfr_opt.predict(X_test)
score_rfr_opt=mean_squared_error(y_test,pred_rfr_opt)
score = rfr_opt.score(X_test, y_test)
print("MSE Random Forest After HyperParameter Tuning With Hyperopt: ", score_rfr_opt)
print("R^2 Score Random Forest After Hyperparameter Tuning With Hyperopt: ", score)

# %%
plt.figure(figsize=(8, 6))
plt.scatter(y_test, pred_rfr_opt, alpha=0.5, label='Predicted', color='blue')
plt.scatter(y_test, y_test, alpha=0.5, label='Actual', color='red')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values (Random Forest - Hyperopt)')
plt.legend()
plt.show()


