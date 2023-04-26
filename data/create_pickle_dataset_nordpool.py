import pandas as pd
import os
import glob
import logging
import shutil
from datetime import datetime
import pathlib


# Testing with elspot folder first. Then we can generalize

def read_csv_wide_hour_format(path, sheet_name, time_zone, area_name='est', header=5):

    ######## TEMPORARY ########
    time_zone = 'Europe/Copenhagen'
    ######## TEMPORARY ########

    # read sdv file with pandas
    df = pd.read_excel(path, sheet_name=sheet_name, header=header)
    # rename first column to datetime
    df = df.rename(columns={'Unnamed: 0': 'datetime'})
    # make date time actual datetime column
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')

    # Now we have columns 1, 2, ... indicating hours of the day. columns 3A and 3B used for daylightsaving
    # first if all 3B is nan, then we have no daylight saving
    # drop all columns with all nan
    df = df.dropna(axis=1, how='all')

    non_numeric_cols = [col for col in df.columns if col not in ['datetime', '3A', '3B'] and type(col) != int]

    # get non numeric
    # drop all columns that are not numeric
    df = df.drop(columns=non_numeric_cols)

    # # melt dataframe
    #area_name = 'est'
    df = pd.melt(df, id_vars=['datetime'], var_name='hour', value_name=area_name + '_price')
    print(len(df))
    df = df.dropna(axis=0, how='any')
    print(len(df))
    daylightsaving_rows = df['hour'] == '3B'
    #replace 3A and 3B with just 3
    df.loc[df['hour'].isin(['3A', '3B']), 'hour'] = 3
    # add hour to datetime # -1 because of starting at 00:00
    df['datetime'] = df['datetime'] + pd.to_timedelta(df['hour'], unit='h') - pd.Timedelta(hours=1)
    # convert to utc
    df['datetime'] = df['datetime'].dt.tz_localize(time_zone, ambiguous=daylightsaving_rows).dt.tz_convert('UTC')
    #df['datetime'] = df['datetime'] # -1 because of starting at 00:00
    # drop hour column
    df = df.drop(columns=['hour'])
    # set datetime as index
    df = df.set_index('datetime')
    # sort index
    df = df.sort_index()

    return df


# lets go
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                        , handlers=[logging.FileHandler('create_pickle_dataset_entsoe.log'), logging.StreamHandler()])

    nordpool_path = 'C:/Users/frede/Nordpool_ftp'
    relevant_folders = ['CWE']
    elspot_path = r'C:\Users\frede\Nordpool_ftp\Elspot\Elspot_prices'

    # Skipping kontek, which is a cable.
    elspot_prices_folders = {'Denmark': {'folders': [{'name': 'Denmark_East' , 'file_prefix': 'cph'},
                                                     {'name': 'Denmark_West', 'file_prefix': 'ode'}]
                                         },
                             'Estonia': {'folders': [], 'file_prefix': 'est-'},
                             'Finland': {'folders': [], 'file_prefix': 'hel'},
                             'Latvia': {'folders': [], 'file_prefix': 'lat-'},
                             'Lithuania': {'folders': [], 'file_prefix': 'lit-'},
                             'System': {'folders': [], 'file_prefix': 'sys'},
                             'Norway': {'folders': [{'name': 'Bergen', 'file_prefix': 'ber-'},
                                                    {'name': 'Kristiansand', 'file_prefix': 'krs-'},
                                                    {'name': 'Kristiansund', 'file_prefix': 'ksund-'},
                                                    {'name': 'Oslo', 'file_prefix': 'os-'},
                                                    {'name': 'Tromso', 'file_prefix': 'tro-'},
                                                    {'name': 'Trondheim', 'file_prefix': 'trh-'}
                                                    ]
                                        },
                             'Sweden': {'folders': [{'name': 'SE1_Lulea', 'file_prefix': 'lul'},
                                                    {'name': 'SE2_Sundsvall', 'file_prefix': 'sund'},
                                                    {'name': 'SE3_Stockholm', 'file_prefix': 'sto'},
                                                    {'name': 'SE4_Malmo', 'file_prefix': 'mal'}
                                                    ]

                                        }
                             }

    # data path
    data_path_str = r'C:\Users\frede\PycharmProjects\Masters\data\data\elspot_prices'
    data_path = pathlib.Path(data_path_str)

    for country, info_dict in elspot_prices_folders.items():
        if country in ['Finland', 'Estonia', 'Latvia', 'Lithuania']:
            tz = 'Europe/Tallinn'
        else:
            tz = 'Europe/Copenhagen'

        if len(info_dict['folders']) == 0:
            # both ends of range are inclusive
            year_list = pd.date_range(start='2015', end='2023', freq='Y', inclusive='left')
            big_df = pd.DataFrame()
            for year in year_list:
                # read file
                # make a path to file with country and year
                file_name = info_dict['file_prefix'] + 'eur' + year.strftime('%y') + '.xls'
                path = os.path.join(elspot_path, country, year.strftime('%Y'), file_name)
                # read file
                print(path)
                df = read_csv_wide_hour_format(path, area_name=country, time_zone=tz, sheet_name=file_name.replace('.xls', ''))
                big_df = pd.concat([big_df, df])
            # save to pickle
            # drop duplicates
            #big_df = big_df[~big_df.index.duplicated(keep='first')]
            big_df.to_pickle(os.path.join(data_path, country + '.pkl'))
        else:
            for folder_dict in info_dict['folders']:
                # both ends of range are inclusive
                year_list = pd.date_range(start='2015', end='2023', freq='Y', inclusive='left')
                big_df = pd.DataFrame()
                for year in year_list:
                    # read file
                    # make a path to file with country and year
                    file_name = folder_dict['file_prefix'] + 'eur' + year.strftime('%y') + '.xls'
                    path = os.path.join(elspot_path, country, folder_dict['name'], year.strftime('%Y'), file_name)
                    # read file
                    print(path)
                    df = read_csv_wide_hour_format(path, area_name=folder_dict['name'], time_zone=tz, sheet_name=file_name.replace('.xls', ''))
                    big_df = pd.concat([big_df, df])
                # save to pickle
                # drop duplicates
                #big_df = big_df[~big_df.index.duplicated(keep='first')]
                big_df.to_pickle(os.path.join(data_path, country + '_' + folder_dict['name'] + '.pkl'))







