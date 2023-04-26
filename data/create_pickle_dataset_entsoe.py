import pandas as pd
import os
import glob
import logging
import shutil
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    , handlers=[logging.FileHandler('create_pickle_dataset_entsoe.log'), logging.StreamHandler()])

entsoe_path = 'C:/Users/frede/Entsoe_SFTP/TP_export'

# start with AcceptedAggregatedOffers_17.1.D
date_range_monthly = pd.date_range(start='2015-01-01', end='2022-12-31', freq='MS')
# folder = 'ActivatedBalancingEnergy_17.1.E'#'AcceptedAggregatedOffers_17.1.D'
# make list of all folders in entsoe_path
folders_path = [folder.replace('\\', '/') for folder in glob.glob(entsoe_path + '/*')]
# get text from last slash to end of string
folders_name = [folder[folder.rfind('/') + 1:] for folder in folders_path]

for i, (path, folder) in enumerate(zip(folders_path, folders_name)):

    big_df = pd.DataFrame()

    for date in date_range_monthly:
        #csv_path = entsoe_path + f"/{folder}/{date.strftime(format='%Y_%m')}_{folder}.csv"
        try:
            df = pd.read_csv(path + f"/{date.strftime(format='%Y_%m')}_{folder}.csv", sep='\t', decimal='.', index_col=0)
            big_df = pd.concat([big_df, df])
        except Exception as e:
            logging.error(e)
            continue
    big_df.index = pd.to_datetime(big_df.index, format='%Y-%m-%d %H:%M:%S.%f')
    # for each col in big_df make a new df with datetime index and col as value
    for col_iter, col in enumerate(big_df.columns):
        df = pd.DataFrame(big_df[col])
        df.index = big_df.index
        #df = df[~df.index.duplicated(keep='first')]
        # make sure folder exists
        if not os.path.exists(f'data/{folder}'):
            os.makedirs(f'data/{folder}')

        df.to_pickle(f'data/{folder}/{col}.pkl')

        logging.info(f'Added {folder}/{col} {datetime.now().strftime("%H:%M:%S")}, [{col_iter+1}/{len(big_df.columns)}]')

    logging.info(f'Added {folder} {datetime.now().strftime("%H:%M:%S")}, [{i+1}/{len(folders_name)}]')
    # remove folder and content in the path
    shutil.rmtree(path)
    logging.info(f'Removed {path} {datetime.now().strftime("%H:%M:%S")}, [{i+1}/{len(folders_name)}]')

# index to datetime

# remove duplicated indexes
#accepted_agg_offers = big_df[~big_df.index.duplicated(keep='first')]
# to pickle
#accepted_agg_offers.to_pickle(f'data/{folder}.pkl')
