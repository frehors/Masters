import pandas as pd
import os
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import streamlit as st

st.set_page_config(page_title="Result Plots", page_icon="📈")

pio.renderers.default = "browser"

# read data
lear_predictions = pd.read_pickle(os.path.join(os.getcwd(), 'results_app', 'lear_preds_all.pkl'))
lear_predictions.index = pd.to_datetime(lear_predictions.index)
actuals = pd.read_pickle(os.path.join(os.getcwd(), 'results_app', 'actuals_all.pkl'))
actuals.index = pd.to_datetime(actuals.index)
actuals.columns = [i for i in range(24)]
# melt data, each column is an hour of the day and index is the date
lear_predictions_melt = pd.melt(lear_predictions.reset_index(), id_vars='index', var_name='hour', value_name='lear')
# actuals id_vars is index value 2
lear_predictions_melt['lear'] = pd.to_numeric(lear_predictions_melt['lear'])
actuals_melt = pd.melt(actuals.reset_index(), id_vars='index', var_name='hour', value_name='actuals')
# remove h from the hour column
lear_predictions_melt['hour'] = lear_predictions_melt['hour'].str.replace('h', '')
# convert hour column to int
lear_predictions_melt['hour'] = pd.to_numeric(lear_predictions_melt['hour'])
actuals_melt['hour'] = pd.to_numeric(actuals_melt['hour'])
# add hour to index column to create a datetime column
lear_predictions_melt['index'] = pd.to_datetime(lear_predictions_melt['index']) + pd.to_timedelta(lear_predictions_melt['hour'], unit='h')
actuals_melt['index'] = pd.to_datetime(actuals_melt['index']) + pd.to_timedelta(actuals_melt['hour'], unit='h')
# drop hour column
lear_predictions_melt.drop('hour', axis=1, inplace=True)
actuals_melt.drop('hour', axis=1, inplace=True)

# merge dataframes
df = pd.merge(lear_predictions_melt, actuals_melt, on=['index'], how='left')
df = df.sort_values(by=['index'])

# make time series plot of actuals vs predictions on streamlit
st.write("## Actuals vs Predictions")
# divide into 8 plots one for each quarter both years
for i in range(8):
    # make figure
    fig = go.Figure()
    # add traces
    year = 2021
    if i >= 4:
        year = 2022
    tmp_df = df[(df['index'].dt.quarter == (i + 1) % 5) & (df['index'].dt.year == year)]
    fig.add_trace(go.Scatter(x=tmp_df.index, y=tmp_df['actuals'], name='Actuals'))
    fig.add_trace(go.Scatter(x=tmp_df.index, y=tmp_df['lear'], name='Predictions'))
    # add layout
    fig.update_layout(title=f"Quarter {i + 1}", xaxis_title="Date", yaxis_title="MWh")
    # add figure to streamlit
    st.plotly_chart(fig, use_container_width=True)
