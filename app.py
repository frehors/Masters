import pandas as pd
import streamlit as st
import plotly.express as px

# get example data using requests
# url = 'https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv'
#
# data = pd.read_csv(url)
#
#
# data_exploration = st.container()
#
# with data_exploration:
#     st.title('Data Exploration')
#     st.subheader('stock data')
#     st.write(data.head())
#     st.markdown('This is the data we will be using for this project')
#     st.markdown('** testing markdown **')
#     st.markdown(r'$$ \mathbb{E} = \sum_{i=0}^\infty \frac{1}{2^i} = 2 $$')
#
#
# # plot close price plotly
#
# close_container = st.container()
#
# with close_container:
#     st.header('Close Price')
#     st.subheader('Aapl stock close')
#     st.text('This is the close price of AAPL stock')
#     close_plot = px.line(data, x='Date', y='AAPL.Close')
#     st.plotly_chart(close_plot)

# setup title and sidebar
st.title('bolig beregner for viktor')
st.sidebar.title('bolig beregner for viktor')

dine_penge = st.sidebar.slider('dine penge', min_value=0, max_value=100000000, value=1000000, step=10000)
bolig_pris = st.sidebar.slider('bolig pris', min_value=0, max_value=10000000, value=1000000, step=10000)
# use slider value to print result

st.write(f'bolig pris: {bolig_pris}, betal {bolig_pris*0.05} kr i depositum')

# write text conditional on slider value
if bolig_pris*0.05 > dine_penge:
    st.write('du har ikke nok penge til depositum')
else:
    st.write('du har nok penge til depositum')




