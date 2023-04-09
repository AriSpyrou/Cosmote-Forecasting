├── Cosmote.ipynb					-> Data preprocessing and random forest classifier
├── Cosmote_Synthetic.ipynb				-> Synthetic data for Nikos/Evie
├── Cosmote_timeseries.ipynb				-> Time series forecasting on tele data
├── Cosmote_timeseries.py				-> Same as above meant to run standalone
├── ICCS - CSLAB Analysis				-> Original data with no mods
│   ├── energy_monitoring				-> energy data
│   │   └── data.csv
│   └── network_stats					-> tele data
│       ├── gsm.csv
│       ├── lte.csv
│       └── nr.csv
├── README.md						-> this file
├── data						-> data after preprocessing
│   ├── cosmote.csv					-> whole dataset
│   ├── cosmote_synthetic_2021.csv
│   ├── cosmote_synthetic_2022.csv
│   ├── cosmote_synthetic_energy.csv
│   ├── synthetic_energy_predictions_2021.csv
│   └── synthetic_energy_predictions_2022.csv
├── lstm.py						-> LSTM NN
├── out							-> output from various sources
│   ├── models.out
│   ├── modelsAIC_enforce.out
│   ├── modelsAIC_noenforce.out
│   ├── noenforce_stationarity_invertibility.out
│   ├── noenforce_stationarity_invertibility_local.out
│   └── output.txt
├── parser.py						-> LSTM input parser
├── pickle						-> pickles from various sources
│   └── rfr.pickle
├── plot_confusion_matrix.py				-> Helper to visualize LSTM performance
└── t_test.ipynb					-> (bad name) Stub to test (S)ARIMA models
