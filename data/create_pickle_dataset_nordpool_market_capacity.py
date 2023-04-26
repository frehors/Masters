import pandas as pd
import os


def read_csv_wide_hour_format_capacities(path, sheet_name, time_zone, header=5):
    # read sdv file with pandas
    df = pd.read_csv(path, sep=';', header=25)  # skiprows=25)
    # add header as first row in the df
    df = pd.concat([df.columns.to_frame().T, df], ignore_index=True)
    # rename columns
    df.columns = header_list
    df['datetime'] = pd.to_datetime(df['Date(dd.mm.yyyy)'], format='%d.%m.%Y')
    df = df.drop(columns=['type', 'Code', 'Year', 'Week', 'Day', 'Date(dd.mm.yyyy)'])
    # make date time actual datetime column

    # rename alias to area
    # dict with 1-24 as keys and 1-24 as values
    rename_dict = {str(i): i for i in range(1, 25)}
    rename_dict['Alias'] = 'market_capacity_area'
    df = df.rename(columns=rename_dict)

    # Now we have columns 1, 2, ... indicating hours of the day. columns 3A and 3B used for daylightsaving
    # first if all 3B is nan, then we have no daylight saving
    # drop all columns with all nan
    #df = df.dropna(axis=1, how='all')

    #non_numeric_cols = [col for col in df.columns if col not in ['datetime', '3A', '3B', 'market_capacity_area'] and type(col) != int]
    #print(non_numeric_cols)
    # get non numeric
    # drop all columns that are not numeric
    df = df.drop(columns=['Sum']) # tror denne er tom, men ak og ve

    # # melt dataframe
    area_name = 'market_capacity_area'
    df = pd.melt(df, id_vars=['datetime', 'market_capacity_area'], var_name='hour', value_name=area_name + '_capacity')
    #print(len(df))
    df = df.dropna(axis=0, how='any')
    #print(len(df))
    daylightsaving_rows = df['hour'] == '3B'
    #replace 3A and 3B with just 3
    df.loc[df['hour'].isin(['3A', '3B']), 'hour'] = 3
    #
    # # add hour to datetime # -1 because of starting at 00:00
    df['datetime'] = df['datetime'] + pd.to_timedelta(df['hour'], unit='h')
    # convert to utc
    df['datetime'] = df['datetime'] - pd.Timedelta(hours=1)  # -1 because of starting at 00:00

    df['datetime'] = df['datetime'].dt.tz_localize(time_zone, ambiguous=daylightsaving_rows).dt.tz_convert('UTC')

    # drop hour column
    df = df.drop(columns=['hour'])
    # make market_capacity_area_capacity numeric, drop non numeric
    df[area_name + '_capacity'] = pd.to_numeric(df[area_name + '_capacity'], errors='coerce')
    df = df.dropna(axis=0, how='any')
    # set datetime as index

    # pivot our market_capacity_area column
    #df = df.set_index('datetime')
    # pivot out market_capacity_area column and keep datetime as index
    df = df.pivot(index='datetime', columns='market_capacity_area', values=area_name + '_capacity')

    #
    # sort index
    #df = df.sort_index()


    return df

if __name__ == '__main__':
    # path to market capacity
    market_capacity_path = r'C:\Users\frede\Nordpool_ftp\Elspot\Market_coupling_capacity'
    data_path = r'C:\Users\frede\PycharmProjects\Masters\data\data\nordpool\market_capacity'
    header_list = ['type', 'Code', 'Year', 'Week', 'Day', 'Date(dd.mm.yyyy)', 'Alias', 'Hour1', 'Hour2', 'Hour3A',
                   'Hour3B', 'Hour4', 'Hour5', 'Hour6', 'Hour7', 'Hour8', 'Hour9', 'Hour10', 'Hour11', 'Hour12',
                   'Hour13', 'Hour14', 'Hour15', 'Hour16', 'Hour17', 'Hour18', 'Hour19', 'Hour20', 'Hour21', 'Hour22',
                   'Hour23', 'Hour24', 'Sum']
    header_list = [x.replace('Hour', '') for x in header_list]

    # make date range from 2015-01-01 to 2022-12-31
    date_range = pd.date_range(start='2015-01-01', end='2022-12-31', freq='Y')
    big_df = pd.DataFrame()
    for year in date_range.year:
        # enter folder
        year_path = os.path.join(market_capacity_path, str(year))
        # get all files in folder
        files = os.listdir(year_path)
        for file in files:
            print(file)
            # get file path
            file_path = os.path.join(year_path, file)
            # read file
            df = read_csv_wide_hour_format_capacities(file_path, 'Sheet1', 'Europe/Copenhagen')
            # join to big df
            big_df = pd.concat([big_df, df])

    # sort index
    big_df = big_df.sort_index()

    # save each column as pickle
    for col in big_df.columns:
        if 'LT' in col:  # We don't need it and it is another time zone
            continue
        df = big_df[col].to_frame()
        # drop all nan
        df = df.dropna(axis=0, how='any')
        df.to_pickle(os.path.join(data_path, col + '.pkl'))


