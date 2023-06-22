import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from epftoolbox.evaluation import DM, plot_multivariate_DM_test, plot_multivariate_GW_test

def sMAPE(y_true, y_pred):
    numerator = 2 * np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    return np.mean(numerator / denominator).round(2)

def rMAE(y_true, y_pred, naive_forecast):
    numerator = np.abs(y_true - y_pred).sum()
    denominator = np.abs(y_true - naive_forecast).sum()
    return round(numerator / denominator, 2)
    #return round(numerator.sum() / denominator.sum(), 2)

os.chdir(r'C:\Users\frede\PycharmProjects\Masters')
actuals = pd.read_pickle(os.path.join(os.getcwd(), 'results_app', 'actuals_all.pkl'))
actuals.index = pd.to_datetime(actuals.index)
actuals.columns = [i for i in range(24)]

naive_forecast = pd.read_pickle(os.path.join(os.getcwd(), 'results_app', 'naive_forecast_all.pkl'))
naive_forecast.index = pd.to_datetime(naive_forecast.index)
naive_forecast.columns = [i for i in range(24)]

dnn_predictions = pd.read_pickle(os.path.join(os.getcwd(), 'results_app', 'dnn4_preds_all.pkl'))
lear_predictions = pd.read_pickle(os.path.join(os.getcwd(), 'results_app', 'lear_preds_all.pkl'))
lstm_predictions = pd.read_pickle(os.path.join(os.getcwd(), 'results_app', 'lstm_preds_all.pkl'))
transformer_predictions = pd.read_pickle(os.path.join(os.getcwd(), 'results_app', 'transformer_preds_all.pkl'))
##### YOu are here

lear_predictions.index = pd.to_datetime(lear_predictions.index)
dnn_predictions.index = pd.to_datetime(dnn_predictions.index)
lstm_predictions.index = pd.to_datetime(lstm_predictions.index)
transformer_predictions.index = pd.to_datetime(transformer_predictions.index)


#melt
dnn_predictions_melt = pd.melt(dnn_predictions.reset_index(), id_vars='index', var_name='hour', value_name='dnn')
lear_predictions_melt = pd.melt(lear_predictions.reset_index(), id_vars='index', var_name='hour', value_name='lear')
lstm_predictions_melt = pd.melt(lstm_predictions.reset_index(), id_vars='index', var_name='hour', value_name='lstm')
transformer_predictions_melt = pd.melt(transformer_predictions.reset_index(), id_vars='index', var_name='hour', value_name='transformer')
actuals_melt = pd.melt(actuals.reset_index(), id_vars='index', var_name='hour', value_name='actuals')
naive_forecast_melt = pd.melt(naive_forecast.reset_index(), id_vars='index', var_name='hour', value_name='naive_forecast')
# remove h from hour column in lear

lear_predictions_melt['hour'] = lear_predictions_melt['hour'].apply(lambda x: x[1:]).astype(int)

# add hour to index column to create a datetime column
dnn_predictions_melt['index'] = pd.to_datetime(dnn_predictions_melt['index']) + pd.to_timedelta(dnn_predictions_melt['hour'], unit='h')
lear_predictions_melt['index'] = pd.to_datetime(lear_predictions_melt['index']) + pd.to_timedelta(lear_predictions_melt['hour'], unit='h')
lstm_predictions_melt['index'] = pd.to_datetime(lstm_predictions_melt['index']) + pd.to_timedelta(lstm_predictions_melt['hour'], unit='h')
transformer_predictions_melt['index'] = pd.to_datetime(transformer_predictions_melt['index']) + pd.to_timedelta(transformer_predictions_melt['hour'], unit='h')
actuals_melt['index'] = pd.to_datetime(actuals_melt['index']) + pd.to_timedelta(actuals_melt['hour'], unit='h')
naive_forecast_melt['index'] = pd.to_datetime(naive_forecast_melt['index']) + pd.to_timedelta(naive_forecast_melt['hour'], unit='h')
# drop hour column
lear_predictions_melt.drop('hour', axis=1, inplace=True)
dnn_predictions_melt.drop('hour', axis=1, inplace=True)
lstm_predictions_melt.drop('hour', axis=1, inplace=True)
transformer_predictions_melt.drop('hour', axis=1, inplace=True)
actuals_melt.drop('hour', axis=1, inplace=True)
naive_forecast_melt.drop('hour', axis=1, inplace=True)
# sort
actuals_melt = actuals_melt.sort_values(by=['index'])
naive_forecast_melt = naive_forecast_melt.sort_values(by=['index'])
dnn_predictions_melt = dnn_predictions_melt.sort_values(by=['index'])
lear_predictions_melt = lear_predictions_melt.sort_values(by=['index'])
lstm_predictions_melt = lstm_predictions_melt.sort_values(by=['index'])
transformer_predictions_melt = transformer_predictions_melt.sort_values(by=['index'])
ensemble_predictions = np.average([dnn_predictions_melt['dnn'], lear_predictions_melt['lear']], axis=0, weights=[0.5, 0.5])
ensemble_predictions_dict = {'index': dnn_predictions_melt['index'], 'ensemble': ensemble_predictions}
ensemble_predictions_melt = pd.DataFrame(ensemble_predictions_dict)
# combine into one df
df = pd.concat([actuals_melt, naive_forecast_melt, dnn_predictions_melt, lear_predictions_melt, lstm_predictions_melt, transformer_predictions_melt, ensemble_predictions_melt], axis=1)
# drop duplicate index column
df = df.loc[:,~df.columns.duplicated()]
# make all columns into float type but not index
# set index
df.set_index('index', inplace=True)
df = df.astype(float)

# drop actuals into a separate df
actuals_df = df[['actuals']]
# drop actuals from df
df.drop('actuals', axis=1, inplace=True)
# apply dm
plot_path = r'C:\Users\frede\PycharmProjects\Masters\plots'
plot_multivariate_GW_test(real_price=actuals_df, forecasts=df, norm=1, savefig=True, path=plot_path)
print()
