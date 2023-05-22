import pandas as pd
import os

def create_dataset(target_col='DK1_price'):
    path = r'C:\Users\frede\PycharmProjects\Masters\data\data\DayAheadGenerationForecastForWindAndSolar_14.1.D\big_df.pkl'

    keep_countries = ['DK1', 'DK2', 'SE', 'NO', 'DE']
    df = pd.read_pickle(path)
    df = df[df['MapCode'].isin(keep_countries)]
    # for now only keep whole hours
    df = df[df['DateTime'].dt.minute == 0]
    # pivot our MapCode and ProductionType
    df = df.pivot_table(index='DateTime', columns=['MapCode', 'ProductionType'], values='AggregatedGenerationForecast')

    # drop SE Solar, since only available from 2022
    df = df.drop(columns=[('SE', 'Solar')])
    valid_start_dt = df[('SE', 'Wind Onshore')].first_valid_index()
    end_dt = df.index[-1]
    df = df[df.index >= valid_start_dt]

    # Combine offshore and onshore wind to lessen the number of features
    df[('DK', 'Wind')] = df[('DK1', 'Wind Onshore')] + df[('DK1', 'Wind Offshore')] + df[('DK2', 'Wind Onshore')] + df[('DK2', 'Wind Offshore')]
    df[('NO', 'Wind')] = df[('NO', 'Wind Onshore')] + df[('NO', 'Wind Offshore')]
    df[('DE', 'Wind')] = df[('DE', 'Wind Onshore')] + df[('DE', 'Wind Offshore')]
    df[('SE', 'Wind')] = df[('SE', 'Wind Onshore')]
    df = df.drop(columns=[('DK1', 'Wind Onshore'), ('DK1', 'Wind Offshore'),
                            ('DK2', 'Wind Onshore'), ('DK2', 'Wind Offshore'),
                            ('NO', 'Wind Onshore'), ('NO', 'Wind Offshore'),
                            ('DE', 'Wind Onshore'), ('DE', 'Wind Offshore'),
                            ('SE', 'Wind Onshore')])



    # make linear interpolation of missing values
    df = df.interpolate(method='linear', axis=0)
    # flatten multiindex
    df.columns = ['_'.join(col) for col in df.columns.values]

    # get prices also
    price_path = r'C:\Users\frede\PycharmProjects\Masters\data\data\elspot_prices'

    tmp = pd.read_pickle(os.path.join(price_path, 'Denmark_Denmark_East.pkl'))

    # price_areas = ['Denmark_Denmark_East.pkl', 'Denmark_Denmark_West.pkl', 'System.pkl',
    #                'Sweden_SE1_Lulea.pkl', 'Sweden_SE2_Sundsvall.pkl', 'Sweden_SE3_Stockholm.pkl',
    #                'Sweden_SE4_Malmo.pkl',
    #                'Norway_Bergen.pkl', 'Norway_Kristiansand.pkl', 'Norway_Oslo.pkl', 'Norway_Tromso.pkl',
    #                'Norway_Trondheim.pkl', 'Norway_Kristiansand.pkl']

    price_areas = ['Denmark_Denmark_West.pkl']


    price_df = pd.DataFrame()
    for file in os.listdir(price_path):

        if file.endswith('.pkl') and file in price_areas:
            tmp_price_df = pd.read_pickle(os.path.join(price_path, file))
            tmp_price_df = tmp_price_df[tmp_price_df.index >= valid_start_dt.strftime(format='%Y-%m-%d %H:%M:%S')]
            tmp_price_df = tmp_price_df[tmp_price_df.index <= end_dt.strftime(format='%Y-%m-%d %H:%M:%S')]
            price_df = pd.concat([tmp_price_df, price_df], axis=1)
    price_df = price_df[~price_df.index.duplicated(keep='first')]
    price_df.index = price_df.index.tz_localize(None)
    price_df = price_df.sort_index()

    ## COMMENTED OUT OTHER COUNTRIES FOR NOW
    # Get average price for countries with multiple zones
    ##price_df['Denmark'] = price_df[['Denmark_Denmark_East', 'Denmark_Denmark_West']].mean(axis=1)



    # price_df['SE_price'] = price_df[['SE1_Lulea_price', 'SE2_Sundsvall_price', 'SE3_Stockholm_price', 'SE4_Malmo_price']].mean(axis=1)
    # price_df['NO_price'] = price_df[['Bergen_price', 'Kristiansand_price', 'Oslo_price', 'Tromso_price',
    #                                  'Trondheim_price', 'Kristiansund_price']].mean(axis=1)
    # # drop the old columns
    # price_df = price_df.drop(columns=['SE1_Lulea_price', 'SE2_Sundsvall_price', 'SE3_Stockholm_price', 'SE4_Malmo_price',
    #                                     'Bergen_price', 'Kristiansand_price', 'Oslo_price', 'Tromso_price',
    #                                   'Trondheim_price', 'Kristiansund_price', 'System_price'])

    ############################


    # rename west denmark to DK1 and east denmark to DK2
    price_df = price_df.rename(columns={'Denmark_East_price': 'DK2_price', 'Denmark_West_price': 'DK1_price'})
    # join price_df and df





    entsoe_path_germany = r'C:\Users\frede\PycharmProjects\Masters\data\data\Entsoe_DayAheadPrices\big_df.pkl'

    entsoe_df = pd.read_pickle(entsoe_path_germany)
    entsoe_df = entsoe_df[entsoe_df['DateTime'] >= valid_start_dt.strftime(format='%Y-%m-%d %H:%M:%S')]
    entsoe_df = entsoe_df[entsoe_df['DateTime'] <= end_dt.strftime(format='%Y-%m-%d %H:%M:%S')]
    entsoe_df = entsoe_df[(entsoe_df['MapCode'] == 'DE_AT_LU') | (entsoe_df['MapCode'] == 'DE_LU')]
    # keep only hourly prices not 15 mins
    entsoe_df = entsoe_df[entsoe_df['ResolutionCode'] == 'PT60M']

    entsoe_df.index = entsoe_df['DateTime']
    entsoe_df = entsoe_df.drop(columns=['DateTime', 'ResolutionCode', 'MapCode'])

    entsoe_df = entsoe_df.rename(columns={'Price': 'DE_price'})

    price_df = pd.merge(price_df, entsoe_df, left_index=True, right_index=True)

    df = pd.merge(df, price_df, left_index=True, right_index=True)
    # make temporal dummy features
    #df['hour'] = df.index.hour
    #df['day_of_week'] = df.index.dayofweek
    #df['month'] = df.index.month        # SKAL SELV TAGE HØJDE FOR DETTE HVIS VI IKKE BRUGER EPFtoolbox
    #df['year'] = df.index.year

    # make lagged features
    lagged_features = {}

    for col in df.columns:
        if col != target_col:
            for i in [1, 2, 3, 7]:
                # lagged series
                lagged_features['{}_lag_{}'.format(col, i)] = df[col].shift(i*24)

    # drop non lagged prices, as these are not available in real time, but keep target_col
    #price_cols = ['DK1_price', 'DK2_price', 'SE_price', 'NO_price', 'DE_price']
    price_cols = ['DK1_price']
    df = df.drop(columns=[col for col in price_cols if col != target_col])


    df = pd.concat([df, pd.DataFrame(lagged_features, index=df.index)], axis=1)
    df['day_of_week'] = df.index.dayofweek

    # to dummies
    # day_of_week_0 column when day_of_week is 0, i.e. monday. 1 if monday, 0 otherwise
    df['day_of_week_0'] = df['day_of_week'].apply(lambda x: 1 if x == 0 else 0)
    df = pd.get_dummies(df, columns=['day_of_week'], drop_first=True) # last one should not be there, but we still use it?

    # drop columns with max = min = 0
    invalid_cols = [col for col in df.columns if df[col].max() == df[col].min()]

    #df.index = df.index.tz_localize('UTC')

    df = df.drop(columns=invalid_cols)
    df = df.dropna()
    df = df.sort_index()

    return df



