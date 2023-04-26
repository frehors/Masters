import os
import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    , handlers=[logging.FileHandler('create_pickle_dataset_entsoe.log'), logging.StreamHandler()])

path = r'C:\Users\frede\Entsoe_SFTP\new_imports\DayAheadGenerationForecastForWindAndSolar_14.1.D'
date_range_monthly = pd.date_range(start='2015-01-01', end='2022-12-31', freq='MS')
folder = 'DayAheadGenerationForecastForWindAndSolar_14.1.D'
keep_cols = ['DateTime', 'MapCode', 'ProductionType', 'AggregatedGenerationForecast']
big_df = pd.DataFrame()
for i, date in enumerate(date_range_monthly):

    df = pd.read_csv(path + f"/{date.strftime(format='%Y_%m')}_{folder}.csv", sep='\t', decimal='.')
    df = df[keep_cols]
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S.%f')
    big_df = pd.concat([big_df, df])

    logging.info(f'Added {folder} {datetime.now().strftime("%H:%M:%S")}, [{i+1}/{len(date_range_monthly)}]')

# save big_df0
save_path = r'C:\Users\frede\PycharmProjects\Masters\data\data\DayAheadGenerationForecastForWindAndSolar_14.1.D'
big_df.to_pickle(save_path + 'big_df.pkl')





