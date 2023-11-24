#!/home/ubuntu/miniconda3/envs/bigoptibase/bin/python
import numpy as np
import pandas as pd
import optuna
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn
from pmdarima.arima import auto_arima

from sklearn.model_selection import KFold, ShuffleSplit, cross_val_score, TimeSeriesSplit

from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from scipy.stats.mstats import winsorize
import warnings

pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")
df = pd.read_csv('data/cosmote.csv', parse_dates=True)
df = df.loc[df.PERIOD_START_TIME > '2022-01-01 23:00:00'].reset_index(drop=True)
df.PERIOD_START_TIME = pd.to_datetime(df.PERIOD_START_TIME)
df = df.drop(columns=['TCH_CONGESTION', 'TCH_BLOCKING', 'AVG_UL_MAC_UE_TPUT'])  # ? -> df.corr().energy_mean.sort_values(ascending=False)

df = df.loc[(df.PERIOD_START_TIME > '2022-01-07 23:00:00') & (df.PERIOD_START_TIME < '2022-03-08')]

df = df.fillna(method='ffill')


basedf = df.copy()
for col in df.columns[3:]:
    df[col] =  winsorize(df[col], limits=[0.01, 0.01])
    df[col] += 1
results = pd.DataFrame()
checkpoint = pd.read_csv('out/interim_sarima_results.csv')
for station, stationdf in df.groupby('ID'):
    stationdf = stationdf.sort_values('PERIOD_START_TIME').reset_index(drop=True)
    print(station)
    # tscv = TimeSeriesSplit(n_splits=2, test_size=24*3)
    # split = list(tscv.split(stationdf))[1]

    # train = stationdf.iloc[split[0]]
    # test = stationdf.iloc[split[1]]

    for i, col in enumerate(stationdf.columns[3:]):
        print(f"Processing {col} for {station}")
        ts = stationdf.loc[:, col]
        tscv = TimeSeriesSplit(n_splits=3, test_size=24*1)
        
        for j, (train_index, test_index) in enumerate(tscv.split(ts)):
            test_exists = checkpoint.loc[(checkpoint.ID == station) & (checkpoint.column == col) & (checkpoint.split  == j+1)].any().sum() > 1
            if test_exists:
                print(f"{station},{col},{j+1} exists cont..")
                continue
            oos = 24
            model = auto_arima(ts.iloc[train_index], \
                               max_p=24, \
                               max_d=2, \
                               max_q=24, \
                               m=24, \
                               max_P=5, \
                               out_of_sample_size=oos, \
                               information_criterion='oob', \
                               stepwise=True, \
                               trace=True)
            params = model.to_dict()
            
            val_index = train_index[-oos:]
            train_index = train_index[:-oos]
            
            train_rmse = mean_squared_error(ts.iloc[train_index], model.fittedvalues()[train_index], squared=False)
            train_mape = mean_absolute_percentage_error(ts.iloc[train_index]+1, model.fittedvalues()[train_index]+1)
            
            val_rmse = mean_squared_error(ts.iloc[val_index], model.fittedvalues()[val_index], squared=False)
            val_mape = mean_absolute_percentage_error(ts.iloc[val_index]+1, model.fittedvalues()[val_index]+1)
            results_line = pd.DataFrame([{'ID': station,
                                        'column': col,
                                        'AR': params['order'][0],
                                        'I': params['order'][1],
                                        'MA': params['order'][2],
                                        'S-AR': params['seasonal_order'][0],
                                        'S-I': params['seasonal_order'][1],
                                        'S-MA': params['seasonal_order'][2],
                                        'train_rmse': train_rmse,
                                        'train_mape': train_mape,
                                        'val_rmse': val_rmse,
                                        'val_mape': val_mape,
                                        'split': j+1
                                        }])

            results = pd.concat([results, results_line], ignore_index=True)

