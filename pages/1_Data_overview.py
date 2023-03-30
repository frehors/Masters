import streamlit as st
import pandas as pd
import glob
st.set_page_config(page_title="Data Overview", page_icon="🌍")

@st.cache_data
def read_data_overview(curve_csv):
    # read csv make first column index using this path: f"data/data_overview/{curve}.csv"
    df = pd.read_csv(f"data/data_overview/{curve_csv}", index_col=0)
    return df

# list of files in data/data_overview
files = [file.replace('\\', '/') for file in glob.glob('data/data_overview/*')]
# make dictionary with folder name as key and list of files as value, # last occurence of /

files = [file[file.rfind('/') + 1:] for file in files]

for file in files:
    # with st.expander("AcceptedAggregatedOffers_17.1.D"):
    #     st.write("This is the data overview page")
    #     st.dataframe(read_data_overview("AcceptedAggregatedOffers_17.1.D"))
    with st.expander(file):
        st.write(f"Overview of {file}")
        st.dataframe(read_data_overview(file))

