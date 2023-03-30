import pandas as pd
import glob
import os
import logging
# for each folder in data/data make an overview of each curve
# and save it to a csv file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    , handlers=[logging.FileHandler('make_data_overview.log'), logging.StreamHandler()])
# get list of folders in data/data
folders = [folder.replace('\\', '/') for folder in glob.glob('data/data/*')]
# make dictionary with folder name as key and list of files as value
folders_dict = {folder[folder.rfind('/') + 1:]: [file.replace('\\', '/') for file in glob.glob(folder + '/*')
                                                 if file.endswith('.pkl')] for folder in folders}


for folder, files in folders_dict.items():
    logging.info(f'Folder: {folder}')
    # datafram of all files in folder
    df = pd.DataFrame()
    for file in files:
        file_df = pd.read_pickle(file)
        # add file name as column name
        file_df.columns = [file[file.rfind('/') + 1:file.rfind('.pkl')]]
        df = pd.concat([df, file_df], axis=1)

    # make summary of dataframe
    df_summary = df.describe()
    # save to csv
    df_summary.to_csv(f'data/data_overview/{folder}.csv')
    logging.info(f'{folder} done! Number of files: {len(files)}')
