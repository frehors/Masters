import pandas as pd
import os
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import streamlit as st
from sklearn.metrics import mean_absolute_error


st.set_page_config(page_title="Result Plots", page_icon="📈")

actuals = pd.read_pickle(os.path.join(os.getcwd(), 'results_app', 'actuals_all.pkl'))
actuals.index = pd.to_datetime(actuals.index)
actuals.columns = [i for i in range(24)]
transformer_predictions = pd.read_pickle(os.path.join(os.getcwd(), 'results_app', 'transformer_preds_all.pkl'))

transformer_predictions.index = pd.to_datetime(transformer_predictions.index)

#melt
transformer_predictions_melt = pd.melt(transformer_predictions.reset_index(), id_vars='index', var_name='hour', value_name='transformer')
actuals_melt = pd.melt(actuals.reset_index(), id_vars='index', var_name='hour', value_name='actuals')

# add hour to index column to create a datetime column
transformer_predictions_melt['index'] = pd.to_datetime(transformer_predictions_melt['index']) + pd.to_timedelta(transformer_predictions_melt['hour'], unit='h')
actuals_melt['index'] = pd.to_datetime(actuals_melt['index']) + pd.to_timedelta(actuals_melt['hour'], unit='h')

# drop hour column
transformer_predictions_melt.drop('hour', axis=1, inplace=True)
actuals_melt.drop('hour', axis=1, inplace=True)

# sort
actuals_melt = actuals_melt.sort_values(by=['index'])
transformer_predictions_melt = transformer_predictions_melt.sort_values(by=['index'])

df = pd.merge(transformer_predictions_melt, actuals_melt, on=['index'], how='left')
df = df.sort_values(by=['index'])
df['error'] = abs(df['actuals'] - df['transformer'])
# make time series plot of actuals vs predictions on streamlit

st.write("## Actuals vs Predictions - transformer")
st.write("### Mean Absolute Error: ", round(mean_absolute_error(df['actuals'], df['transformer']), 2))
#col1, col2 = st.columns([4, 1])

# monthly basis
for i in range(2 * 12):
    # make figure

    # add traces
    year = 2021
    month = pd.to_datetime((i % 12) + 1, format='%m').month_name()

    if i >= 12:
        year = 2022
    tmp_df = df[(df['index'].dt.month == (i % 12) + 1) & (df['index'].dt.year == year)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tmp_df['index'], y=tmp_df['actuals'], name='Actuals'))
    fig.add_trace(go.Scatter(x=tmp_df['index'], y=tmp_df['transformer'], name='Predictions'))
    # add layout
    # convert number to month
    fig.update_layout(title=f"{year} {month}", xaxis_title="Date", yaxis_title="€/MWh")
    # write the MAE for the month on the figure
    fig.add_annotation(x=0.5, y=0.9, xref="paper", yref="paper",
                          text=f"MAE: {round(mean_absolute_error(tmp_df['actuals'], tmp_df['transformer']), 2)}",
                            showarrow=False)
    # add figure to streamlit
    #with st.expander(f"Year {year} Quarter {i + 1}"):
    st.plotly_chart(fig, use_container_width=True)