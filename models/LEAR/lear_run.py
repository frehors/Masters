import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from models.create_dataset import create_dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# lasso model
from epftoolbox.models import LEAR
from sklearn.linear_model import Lasso
import logging
import os
import time
# import MAE
from sklearn.metrics import mean_absolute_error


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    , handlers=[logging.FileHandler('LEAR_epf.log'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# load data
target_col = 'DK1_price'
df = create_dataset(target_col=target_col)
# split into X and y
y = df[target_col]
X = df.drop(target_col, axis=1)
# Pivot hourly index out to columns so index is only date
pivot_columns = [col for col in X.columns if not col.startswith('day_of_week')]
X = X.pivot_table(index=X.index.date, columns=X.index.hour, values=pivot_columns)
X = X.dropna()
# add day of week as feature

# Some hours will only have 0 values, drop these columns (e.g. Solar)
X = X.loc[:, (X != 0).any(axis=0)]
# and some are 0 almost always, drop features with a MAD below threshold
X = X.loc[:, X.sub(X.median(axis=0), axis=1).abs().median(axis=0) > 0.01]

X.index = pd.to_datetime(X.index)
X['day_of_week'] = X.index.dayofweek

# to dummies
# day_of_week_0 column when day_of_week is 0, i.e. monday. 1 if monday, 0 otherwise
X['day_of_week_0'] = X['day_of_week'].apply(lambda x: 1 if x == 0 else 0)
X = pd.get_dummies(X, columns=['day_of_week'], drop_first=True) # last one should not be there, but we still use it?
# apparently all variables need to be named Exogenous 1, 2, 3, etc.
X.columns = ['Exogenous ' + str(i) for i in range(1, X.shape[1] + 1)]

# now y
# make y to dataframe first
y = y.rename('Price')
y = pd.DataFrame(y)
y = y.pivot_table(index=y.index.date, columns=y.index.hour, values=y.columns)
# join multiindex columns to one, price with hour number
y = y.dropna()
# rename to 'Price' for some reason for this toolbox to work



##### temporary start cut off as well to see if it works
#start_cutoff = pd.to_datetime('2019-01-01 00:00')
val_cutoff = pd.to_datetime('2020-07-01')
test_cutoff = pd.to_datetime('2021-01-01')
#X_train = X.loc[X.index >= start_cutoff & X.index < val_cutoff]
X_train = X.loc[X.index < val_cutoff]
X_val = X.loc[(X.index >= val_cutoff) & (X.index < test_cutoff)]
X_test = X.loc[X.index >= test_cutoff]
#y_train = y.loc[y.index < val_cutoff & y.index >= start_cutoff]
y_train = y.loc[y.index < val_cutoff]
y_val = y.loc[(y.index >= val_cutoff) & (y.index < test_cutoff)]
y_test = y.loc[y.index >= test_cutoff]

# split into train and test
# set start time as first time in index where hour is 0
#
# start_time = X[X.index.hour == 0].index[0]
# test_cutoff = pd.to_datetime('2021-01-01 00:00')
# X = X[X.index >= start_time]
# y = y[y.index >= start_time]
#
# # subtract one hour
# X_train, X_test = X[X.index < test_cutoff], X[X.index >= test_cutoff]
# y_train, y_test = y[y.index < test_cutoff], y[y.index >= test_cutoff]
# split into train and validation
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

#pd.to_datetime('2021-01-01 00:00')
#%%
# fit Lear model
calibration_window = 365 * 2  # 2 years
#day_range = pd.date_range(start=test_cutoff, end=y.index[-1], freq='D')
day_range = pd.date_range(start=pd.to_datetime('2021-01-01'), end=pd.to_datetime('2022-12-31'), freq='D')
#preds = pd.DataFrame(index=y_test.index, columns=['pred'])
#preds_done = pd.read_pickle(os.path.join(os.getcwd(), 'predictions', 'lear_preds_all.pkl'))
forecast = pd.read_pickle(os.path.join(os.getcwd(), 'predictions', 'lear_preds_2021.pkl'))
#forecast = pd.DataFrame(index=X_test.index, columns=['h' + str(k) for k in range(24)])
forecast.index = pd.to_datetime(forecast.index)
#preds_done = preds_done.dropna()
#preds_done.index = pd.to_datetime(preds_done.index)
# Acess first element of list for each row in preds_done
# for i, date in enumerate(preds_done.index):
#     forecast.loc[date] = preds_done.iloc[i]['pred'][0]#forecast.loc[preds_done.index] = preds_done.values.reshape(-1, 1)

model = LEAR(calibration_window=calibration_window)

# rename to 'Price' for some reason for this toolbox to work
target_col = 'Price'


time_start = time.time()
for i, day in enumerate(day_range):
    if day.day == 1:
        logger.info(f'Predicting day {day} ({i+1}/{len(day_range)})')
        # save predictions
        forecast.to_pickle(os.path.join(os.getcwd(), 'predictions', 'lear_preds_2.pkl'))
    # get train data
    X_train_date = X[X.index < day].values
    y_train_date = y[y.index < day].values

    X_test_date = X[(X.index >= day) & (X.index < day + pd.Timedelta(days=1))].values

    pred = model.recalibrate_predict(X_train_date, y_train_date, X_test_date)

    forecast.loc[day] = pred
    print(f'MAE: {mean_absolute_error(y_test[y_test.index == day.date()], pred)}')
    # time remaining
    time_elapsed = time.time() - time_start
    time_per_day = time_elapsed / (i + 1)
    days_remaining = len(day_range) - (i + 1)
    time_remaining = time_per_day * days_remaining
    print(f'Time remaining: {time_remaining / 3600} hours')

print(f'Time elapsed: {(time.time() - time_start)/3600} hours')
forecast.to_pickle(os.path.join(os.getcwd(), 'predictions', 'lear_preds_2.pkl'))



