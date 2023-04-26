import pandas as pd
import os

path = r'C:\Users\frede\PycharmProjects\Masters\data\data\DayAheadGenerationForecastForWindAndSolar_14.1.D'

df = pd.DataFrame()
for file in os.listdir(path):
    if file.endswith('.pkl'):
        df = pd.concat([df, pd.read_pickle(os.path.join(path, file))], axis=1)
df = df.sort_index()
# denmark and neighboring countries list
relevant_countries = ['DK1', 'DK2', 'DE_TenneT_GER', 'DE_Amprion', 'DE_50HzT', 'DE_TransnetBW',
                      'SE1', 'SE2', 'SE3', 'SE4', 'NO1', 'NO2', 'NO3', 'NO4', 'NO5']
# only keep rows where the value in MapCode is in relevant countries list
df = df[df['MapCode'].isin(relevant_countries)]

keep_cols = ['AggregatedGenerationForecast', 'MapCode', 'ProductionType']
df = df[keep_cols]

# pivot out MapCode and ProductionType
#
#df = df.pivot_table(index=df.index, columns=['MapCode', 'ProductionType'], values='AggregatedGenerationForecast')
#
# for col in df.columns:
#     curve = df[col]
#     curve = curve.dropna()
#     curve = curve.sort_index()
#     # save to pickle
#     print(col, curve.shape)
#     if curve.shape[0] > 0:
#         save_path = os.path.join(path, 'countries', col[0] + '_' + col[1] + '.pkl')
#         curve.to_pickle(save_path)
