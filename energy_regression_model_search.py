#!/home/ubuntu/miniconda3/envs/bigoptibase/bin/python
import numpy as np
import pandas as pd
import optuna
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn

from sklearn.model_selection import KFold, ShuffleSplit, cross_val_score

from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from scipy.stats.mstats import winsorize


df = pd.read_csv('data/cosmote.csv', parse_dates=True)
df = df.fillna(method='ffill')
df['energy_mean_base'] = df['energy_mean'].copy()
df['energy_mean'] = winsorize(df['energy_mean'], limits=[0.05, 0.05])
df = df.drop(columns=['TCH_CONGESTION', 'TCH_BLOCKING', 'AVG_UL_MAC_UE_TPUT'])  # ? -> df.corr().energy_mean.sort_values(ascending=False

random_split = list(ShuffleSplit(n_splits=1, test_size=0.1, random_state=1).split(df))[0]
train_index = random_split[0]
test_index = random_split[1]

trainDf = df.loc[train_index, ~df.columns.isin(['ID', 'PERIOD_START_TIME', 'energy_mean', 'energy_mean_base'])]

random_split = list(ShuffleSplit(n_splits=1, test_size=0.2, random_state=1).split(trainDf))[0]
train_index = random_split[0]
val_index = random_split[1]

trainDf = df.loc[train_index, ~df.columns.isin(['ID', 'PERIOD_START_TIME', 'energy_mean', 'energy_mean_base'])]
y_train = df.loc[train_index, 'energy_mean']
train_ndarray = np.array(trainDf)
y_train_ndarray = np.array(y_train)

valDf = df.loc[val_index, ~df.columns.isin(['ID', 'PERIOD_START_TIME', 'energy_mean', 'energy_mean_base'])]
y_val = df.loc[val_index, 'energy_mean']
val_ndarray = np.array(valDf)
y_val_ndarray = np.array(y_val)

testDf = df.loc[test_index, ~df.columns.isin(['ID', 'PERIOD_START_TIME', 'energy_mean', 'energy_mean_base'])]
y_test = df.loc[test_index, 'energy_mean']
test_ndarray = np.array(testDf)
y_test_ndarray = np.array(y_test)

def objective(trial):
    global train_ndarray
    global y_train_ndarray
    global y_val_ndarray

    classifier_name = trial.suggest_categorical("classifier", ["SVR", "RandomForest", "ElasticNet", "KNeighbors"])
    if classifier_name == "SVR":
        kernel = trial.suggest_categorical('kernel', ["rbf"])
        tol_svr = trial.suggest_float('tol_svr', 1e-3, 10, log=True)
        c = trial.suggest_float("c", 1e-1, 1e4, log=True)
        classifier_obj = sklearn.svm.SVR(C=c, kernel=kernel, tol=tol_svr)
    elif classifier_name == "RandomForest":
        max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
        criterion = trial.suggest_categorical("criterion", ['squared_error', 'friedman_mse'])
        max_features = trial.suggest_categorical("max_features", ['sqrt', 'log2', None])
        bootstrap = trial.suggest_categorical("bootstrap", [True, False])
        n_estimators = trial.suggest_int("n_estimators", 10, 760, 50)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 32, log=True)
        min_samples_split = trial.suggest_int("min_samples_split", 5, 50, 5)
        
        classifier_obj = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, criterion=criterion, max_features=max_features, bootstrap=bootstrap, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
    elif classifier_name == "KNeighbors":
        n_neighbors = trial.suggest_int("n_neighbors", 3, 19, 2)
        weights = trial.suggest_categorical("weights", ['uniform', 'distance'])
        
        classifier_obj = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
    elif classifier_name == "ElasticNet":
        alpha = trial.suggest_float("alpha", 0.1, 1)
        l1_ratio = trial.suggest_float("l1_ratio", 0, 1)
        tol_en = trial.suggest_float("tol_en", 1e-3, 1)
        selection = trial.suggest_categorical("selection", ['cyclic', 'random'])
        
        classifier_obj = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, tol=tol_en, selection=selection, precompute=True)
    
    model = classifier_obj.fit(train_ndarray, y_train_ndarray)
    preds = model.predict(val_ndarray)
    train_preds = model.predict(train_ndarray)
    
    train_preds = model.predict(train_ndarray)
    train_rmse = mean_squared_error(y_train_ndarray, train_preds, squared=False)
    train_mape = mean_absolute_percentage_error(y_train_ndarray, train_preds)
    
    val_rmse = mean_squared_error(y_val_ndarray, preds, squared=False)
    val_mape = mean_absolute_percentage_error(y_val_ndarray, preds)
    
    return val_rmse, val_mape, train_rmse, train_mape

study = optuna.create_study(directions=['minimize', 'minimize', 'minimize', 'minimize'])
study.optimize(objective, n_trials=1000, n_jobs=-1, gc_after_trial=True, timeout=3600)

trials = study.trials_dataframe()
trials.to_csv('out/models_energy_regression.csv')
