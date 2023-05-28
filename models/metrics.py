import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

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

# drop index column on all but actuals column
# dnn_predictions_melt.drop('index', axis=1, inplace=True)
# lear_predictions_melt.drop('index', axis=1, inplace=True)
# lstm_predictions_melt.drop('index', axis=1, inplace=True)
# transformer_predictions_melt.drop('index', axis=1, inplace=True)

# compute metrics

# lear
lear_mae = round(mean_absolute_error(actuals_melt['actuals'], lear_predictions_melt['lear']), 2)
lear_rmse = round(mean_squared_error(actuals_melt['actuals'], lear_predictions_melt['lear'])**0.5, 2)
lear_mape = round(mean_absolute_percentage_error(actuals_melt['actuals'], lear_predictions_melt['lear']), 2)
lear_smape = sMAPE(actuals_melt['actuals'], lear_predictions_melt['lear'])
lear_rmae = rMAE(actuals_melt['actuals'], lear_predictions_melt['lear'], naive_forecast_melt['naive_forecast'])

print('lear mae: ', lear_mae)
print('lear rmse: ', lear_rmse)
print('lear mape: ', lear_mape)
print('lear smape: ', lear_smape)
print('lear rmae: ', lear_rmae)
# dnn
dnn_mae = round(mean_absolute_error(actuals_melt['actuals'], dnn_predictions_melt['dnn']), 2)
dnn_rmse = round(mean_squared_error(actuals_melt['actuals'], dnn_predictions_melt['dnn'])**0.5, 2)
dnn_mape = round(mean_absolute_percentage_error(actuals_melt['actuals'], dnn_predictions_melt['dnn']), 2)
dnn_smape = sMAPE(actuals_melt['actuals'], dnn_predictions_melt['dnn'])
dnn_rmae = rMAE(actuals_melt['actuals'], dnn_predictions_melt['dnn'], naive_forecast_melt['naive_forecast'])

print('dnn mae: ', dnn_mae)
print('dnn rmse: ', dnn_rmse)
print('dnn mape: ', dnn_mape)
print('dnn smape: ', dnn_smape)
print('dnn rmae: ', dnn_rmae)
# lstm
lstm_mae = round(mean_absolute_error(actuals_melt['actuals'], lstm_predictions_melt['lstm']), 2)
lstm_rmse = round(mean_squared_error(actuals_melt['actuals'], lstm_predictions_melt['lstm'])**0.5, 2)
lstm_mape = round(mean_absolute_percentage_error(actuals_melt['actuals'], lstm_predictions_melt['lstm']), 2)
lstm_smape = sMAPE(actuals_melt['actuals'], lstm_predictions_melt['lstm'])
lstm_rmae = rMAE(actuals_melt['actuals'], lstm_predictions_melt['lstm'], naive_forecast_melt['naive_forecast'])

print('lstm mae: ', lstm_mae)
print('lstm rmse: ', lstm_rmse)
print('lstm mape: ', lstm_mape)
print('lstm smape: ', lstm_smape)
print('lstm rmae: ', lstm_rmae)
# transformer
transformer_mae = round(mean_absolute_error(actuals_melt['actuals'], transformer_predictions_melt['transformer']), 2)
transformer_rmse = round(mean_squared_error(actuals_melt['actuals'], transformer_predictions_melt['transformer'])**0.5, 2)
transformer_mape = round(mean_absolute_percentage_error(actuals_melt['actuals'], transformer_predictions_melt['transformer']), 2)
transformer_smape = sMAPE(actuals_melt['actuals'], transformer_predictions_melt['transformer'])
transformer_rmae = rMAE(actuals_melt['actuals'], transformer_predictions_melt['transformer'], naive_forecast_melt['naive_forecast'])

print('transformer mae: ', transformer_mae)
print('transformer rmse: ', transformer_rmse)
print('transformer mape: ', transformer_mape)
print('transformer smape: ', transformer_smape)
print('transformer rmae: ', transformer_rmae)

# naive

naive_mae = round(mean_absolute_error(actuals_melt['actuals'], naive_forecast_melt['naive_forecast']), 2)
naive_rmse = round(mean_squared_error(actuals_melt['actuals'], naive_forecast_melt['naive_forecast'])**0.5, 2)
naive_mape = round(mean_absolute_percentage_error(actuals_melt['actuals'], naive_forecast_melt['naive_forecast']), 2)
naive_smape = sMAPE(actuals_melt['actuals'], naive_forecast_melt['naive_forecast'])
naive_rmae = rMAE(actuals_melt['actuals'], naive_forecast_melt['naive_forecast'], naive_forecast_melt['naive_forecast'])

print('naive mae: ', naive_mae)
print('naive rmse: ', naive_rmse)
print('naive mape: ', naive_mape)
print('naive smape: ', naive_smape)
print('naive rmae: ', naive_rmae)

# now do Before Energy crisis and After Energy crisis
# Before Energy crisis
# until 2021-09-01

crisis_cutoff = pd.to_datetime('2021-09-01')

actuals_melt_before = actuals_melt[actuals_melt['index'] < crisis_cutoff]
naive_forecast_melt_before = naive_forecast_melt[naive_forecast_melt['index'] < crisis_cutoff]
lear_predictions_melt_before = lear_predictions_melt[lear_predictions_melt['index'] < crisis_cutoff]
dnn_predictions_melt_before = dnn_predictions_melt[dnn_predictions_melt['index'] < crisis_cutoff]
lstm_predictions_melt_before = lstm_predictions_melt[lstm_predictions_melt['index'] < crisis_cutoff]
transformer_predictions_melt_before = transformer_predictions_melt[transformer_predictions_melt['index'] < crisis_cutoff]

# lear
lear_mae_before = round(mean_absolute_error(actuals_melt_before['actuals'], lear_predictions_melt_before['lear']), 2)
lear_rmse_before = round(mean_squared_error(actuals_melt_before['actuals'], lear_predictions_melt_before['lear'])**0.5, 2)
lear_mape_before = round(mean_absolute_percentage_error(actuals_melt_before['actuals'], lear_predictions_melt_before['lear']), 2)
lear_smape_before = sMAPE(actuals_melt_before['actuals'], lear_predictions_melt_before['lear'])
lear_rmae_before = rMAE(actuals_melt_before['actuals'], lear_predictions_melt_before['lear'], naive_forecast_melt_before['naive_forecast'])

print('lear mae before: ', lear_mae_before)
print('lear rmse before: ', lear_rmse_before)
print('lear mape before: ', lear_mape_before)
print('lear smape before: ', lear_smape_before)
print('lear rmae before: ', lear_rmae_before)
# dnn
dnn_mae_before = round(mean_absolute_error(actuals_melt_before['actuals'], dnn_predictions_melt_before['dnn']), 2)
dnn_rmse_before = round(mean_squared_error(actuals_melt_before['actuals'], dnn_predictions_melt_before['dnn'])**0.5, 2)
dnn_mape_before = round(mean_absolute_percentage_error(actuals_melt_before['actuals'], dnn_predictions_melt_before['dnn']), 2)
dnn_smape_before = sMAPE(actuals_melt_before['actuals'], dnn_predictions_melt_before['dnn'])
dnn_rmae_before = rMAE(actuals_melt_before['actuals'], dnn_predictions_melt_before['dnn'], naive_forecast_melt_before['naive_forecast'])

print('dnn mae before: ', dnn_mae_before)
print('dnn rmse before: ', dnn_rmse_before)
print('dnn mape before: ', dnn_mape_before)
print('dnn smape before: ', dnn_smape_before)
print('dnn rmae before: ', dnn_rmae_before)
# lstm
lstm_mae_before = round(mean_absolute_error(actuals_melt_before['actuals'], lstm_predictions_melt_before['lstm']), 2)
lstm_rmse_before = round(mean_squared_error(actuals_melt_before['actuals'], lstm_predictions_melt_before['lstm'])**0.5, 2)
lstm_mape_before = round(mean_absolute_percentage_error(actuals_melt_before['actuals'], lstm_predictions_melt_before['lstm']), 2)
lstm_smape_before = sMAPE(actuals_melt_before['actuals'], lstm_predictions_melt_before['lstm'])
lstm_rmae_before = rMAE(actuals_melt_before['actuals'], lstm_predictions_melt_before['lstm'], naive_forecast_melt_before['naive_forecast'])

print('lstm mae before: ', lstm_mae_before)
print('lstm rmse before: ', lstm_rmse_before)
print('lstm mape before: ', lstm_mape_before)
print('lstm smape before: ', lstm_smape_before)
print('lstm rmae before: ', lstm_rmae_before)

# transformer
transformer_mae_before = round(mean_absolute_error(actuals_melt_before['actuals'], transformer_predictions_melt_before['transformer']), 2)
transformer_rmse_before = round(mean_squared_error(actuals_melt_before['actuals'], transformer_predictions_melt_before['transformer'])**0.5, 2)
transformer_mape_before = round(mean_absolute_percentage_error(actuals_melt_before['actuals'], transformer_predictions_melt_before['transformer']), 2)
transformer_smape_before = sMAPE(actuals_melt_before['actuals'], transformer_predictions_melt_before['transformer'])
transformer_rmae_before = rMAE(actuals_melt_before['actuals'], transformer_predictions_melt_before['transformer'], naive_forecast_melt_before['naive_forecast'])

print('transformer mae before: ', transformer_mae_before)
print('transformer rmse before: ', transformer_rmse_before)
print('transformer mape before: ', transformer_mape_before)
print('transformer smape before: ', transformer_smape_before)
print('transformer rmae before: ', transformer_rmae_before)

# naive

naive_mae_before = round(mean_absolute_error(actuals_melt_before['actuals'], naive_forecast_melt_before['naive_forecast']), 2)
naive_rmse_before = round(mean_squared_error(actuals_melt_before['actuals'], naive_forecast_melt_before['naive_forecast'])**0.5, 2)
naive_mape_before = round(mean_absolute_percentage_error(actuals_melt_before['actuals'], naive_forecast_melt_before['naive_forecast']), 2)
naive_smape_before = sMAPE(actuals_melt_before['actuals'], naive_forecast_melt_before['naive_forecast'])
naive_rmae_before = rMAE(actuals_melt_before['actuals'], naive_forecast_melt_before['naive_forecast'], naive_forecast_melt_before['naive_forecast'])

print('naive mae before: ', naive_mae_before)
print('naive rmse before: ', naive_rmse_before)
print('naive mape before: ', naive_mape_before)
print('naive smape before: ', naive_smape_before)
print('naive rmae before: ', naive_rmae_before)

#after crisis, define after
actuals_melt_after = actuals_melt[actuals_melt['index'] >= crisis_cutoff]
lear_predictions_melt_after = lear_predictions_melt[lear_predictions_melt['index'] >= crisis_cutoff]
dnn_predictions_melt_after = dnn_predictions_melt[dnn_predictions_melt['index'] >= crisis_cutoff]
lstm_predictions_melt_after = lstm_predictions_melt[lstm_predictions_melt['index'] >= crisis_cutoff]
transformer_predictions_melt_after = transformer_predictions_melt[transformer_predictions_melt['index'] >= crisis_cutoff]
naive_forecast_melt_after = naive_forecast_melt[naive_forecast_melt['index'] >= crisis_cutoff]

#lear
lear_mae_after = round(mean_absolute_error(actuals_melt_after['actuals'], lear_predictions_melt_after['lear']), 2)
lear_rmse_after = round(mean_squared_error(actuals_melt_after['actuals'], lear_predictions_melt_after['lear'])**0.5, 2)
lear_mape_after = round(mean_absolute_percentage_error(actuals_melt_after['actuals'], lear_predictions_melt_after['lear']), 2)
lear_smape_after = sMAPE(actuals_melt_after['actuals'], lear_predictions_melt_after['lear'])
lear_rmae_after = rMAE(actuals_melt_after['actuals'], lear_predictions_melt_after['lear'], naive_forecast_melt_after['naive_forecast'])

print('lear mae after: ', lear_mae_after)
print('lear rmse after: ', lear_rmse_after)
print('lear mape after: ', lear_mape_after)
print('lear smape after: ', lear_smape_after)
print('lear rmae after: ', lear_rmae_after)
# dnn
dnn_mae_after = round(mean_absolute_error(actuals_melt_after['actuals'], dnn_predictions_melt_after['dnn']), 2)
dnn_rmse_after = round(mean_squared_error(actuals_melt_after['actuals'], dnn_predictions_melt_after['dnn'])**0.5, 2)
dnn_mape_after = round(mean_absolute_percentage_error(actuals_melt_after['actuals'], dnn_predictions_melt_after['dnn']), 2)
dnn_smape_after = sMAPE(actuals_melt_after['actuals'], dnn_predictions_melt_after['dnn'])
dnn_rmae_after = rMAE(actuals_melt_after['actuals'], dnn_predictions_melt_after['dnn'], naive_forecast_melt_after['naive_forecast'])

print('dnn mae after: ', dnn_mae_after)
print('dnn rmse after: ', dnn_rmse_after)
print('dnn mape after: ', dnn_mape_after)
print('dnn smape after: ', dnn_smape_after)
print('dnn rmae after: ', dnn_rmae_after)

# lstm
lstm_mae_after = round(mean_absolute_error(actuals_melt_after['actuals'], lstm_predictions_melt_after['lstm']), 2)
lstm_rmse_after = round(mean_squared_error(actuals_melt_after['actuals'], lstm_predictions_melt_after['lstm'])**0.5, 2)
lstm_mape_after = round(mean_absolute_percentage_error(actuals_melt_after['actuals'], lstm_predictions_melt_after['lstm']), 2)
lstm_smape_after = sMAPE(actuals_melt_after['actuals'], lstm_predictions_melt_after['lstm'])
lstm_rmae_after = rMAE(actuals_melt_after['actuals'], lstm_predictions_melt_after['lstm'], naive_forecast_melt_after['naive_forecast'])

print('lstm mae after: ', lstm_mae_after)
print('lstm rmse after: ', lstm_rmse_after)
print('lstm mape after: ', lstm_mape_after)
print('lstm smape after: ', lstm_smape_after)
print('lstm rmae after: ', lstm_rmae_after)

# transformer
transformer_mae_after = round(mean_absolute_error(actuals_melt_after['actuals'], transformer_predictions_melt_after['transformer']), 2)
transformer_rmse_after = round(mean_squared_error(actuals_melt_after['actuals'], transformer_predictions_melt_after['transformer'])**0.5, 2)
transformer_mape_after = round(mean_absolute_percentage_error(actuals_melt_after['actuals'], transformer_predictions_melt_after['transformer']), 2)
transformer_smape_after = sMAPE(actuals_melt_after['actuals'], transformer_predictions_melt_after['transformer'])
transformer_rmae_after = rMAE(actuals_melt_after['actuals'], transformer_predictions_melt_after['transformer'], naive_forecast_melt_after['naive_forecast'])

print('transformer mae after: ', transformer_mae_after)
print('transformer rmse after: ', transformer_rmse_after)
print('transformer mape after: ', transformer_mape_after)
print('transformer smape after: ', transformer_smape_after)
print('transformer rmae after: ', transformer_rmae_after)

# naive
naive_mae_after = round(mean_absolute_error(actuals_melt_after['actuals'], naive_forecast_melt_after['naive_forecast']), 2)
naive_rmse_after = round(mean_squared_error(actuals_melt_after['actuals'], naive_forecast_melt_after['naive_forecast'])**0.5, 2)
naive_mape_after = round(mean_absolute_percentage_error(actuals_melt_after['actuals'], naive_forecast_melt_after['naive_forecast']), 2)
naive_smape_after = sMAPE(actuals_melt_after['actuals'], naive_forecast_melt_after['naive_forecast'])
naive_rmae_after = rMAE(actuals_melt_after['actuals'], naive_forecast_melt_after['naive_forecast'], naive_forecast_melt_after['naive_forecast'])

print('naive mae after: ', naive_mae_after)
print('naive rmse after: ', naive_rmse_after)
print('naive mape after: ', naive_mape_after)
print('naive smape after: ', naive_smape_after)
print('naive rmae after: ', naive_rmae_after)





