from models.create_dataset import create_dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import hyperopt
import pickle
import time

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    , handlers=[logging.FileHandler('DNN.log'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create dataset
target_col = 'DK1_price'
df = create_dataset(target_col)

y = df[target_col]
X = df.drop(target_col, axis=1)

# Pivot hourly index out to columns so index is only date
pivot_columns = [col for col in X.columns if not col.startswith('day_of_week')]
X = X.pivot_table(index=X.index.date, columns=X.index.hour, values=pivot_columns)
X.columns = [f'hour_{col}' for col in X.columns]
X = X.dropna()
# Some hours will only have 0 values, drop these columns (e.g. Solar)
X = X.loc[:, (X != 0).any(axis=0)]
# and some are 0 almost always, drop features with a MAD below threshold
X = X.loc[:, X.sub(X.median(axis=0), axis=1).abs().median(axis=0) > 0.01]

# save median and mad for inverse transform
#X_median = X.median(axis=0)
#X_mad = X.mad(axis=0)


#X.sub(median, axis=1)
#X.div(mad, axis=1)

#

# Invariant asinh transform of data
# first Subtract median and then divide by median absolute deviation
class Scaler():

    def __init__(self, median=None, mad=None):
        self.median = None
        self.mad = None

    def fit(self, data):
        if isinstance(data, pd.Series):
            data = pd.DataFrame(data)
        self.median = data.median(axis=0).to_numpy()
        # calculate median absolute deviation
        self.mad = data.sub(data.median(axis=0), axis=1).abs().median(axis=0).to_numpy()
        # print na in mad
        return self

    def transform(self, data):
        if self.median is None or self.mad is None:
            raise ValueError('Fit scaler first')

        if isinstance(data, pd.Series):
            data = pd.DataFrame(data)
        X_transformed = data.sub(self.median, axis=1)
        X_transformed = X_transformed.div(self.mad, axis=1)
        X_transformed = np.arcsinh(X_transformed)

        return X_transformed

    def inverse_transform(self, data):

        if self.median is None or self.mad is None:
            raise ValueError('Fit scaler first')
        # fix so this works for series and dataframe
        if isinstance(data, pd.Series):
            data = pd.DataFrame(data)

        X_inversed = np.sinh(data)
        X_inversed = X_inversed.mul(self.mad, axis=1)
        X_inversed = X_inversed.add(self.median, axis=1)
        # make this work for series


        return X_inversed



#X = transform(X)
X.index = pd.to_datetime(X.index)
X['day_of_week'] = X.index.dayofweek

# to dummies
# day_of_week_0 column when day_of_week is 0, i.e. monday. 1 if monday, 0 otherwise
X['day_of_week_0'] = X['day_of_week'].apply(lambda x: 1 if x == 0 else 0)
X = pd.get_dummies(X, columns=['day_of_week'], drop_first=True) # last one should not be there, but we still use it?

# Drop Nan rows ( should be only first because im running UTC and thus the first "day" doesn't have 24 hrs)
X.isna().sum().sum()

# now y
# make y to dataframe first, should alreadt be, but just to be sure
y = pd.DataFrame(y)
y = y.pivot_table(index=y.index.date, columns=y.index.hour, values=y.columns)
# join multiindex columns to one, price with hour number
y.index = pd.to_datetime(y.index)

y = y.dropna()

##### temporary start cut off as well to see if it works
#start_cutoff = pd.to_datetime('2019-01-01 00:00')
val_cutoff = pd.to_datetime('2020-07-01')
test_cutoff = pd.to_datetime('2021-01-01')

# naive_cutoff = test_cutoff - pd.Timedelta(days=7)
# y_naive = y.loc[y.index >= naive_cutoff]
#
# y_naive = y_naive.shift(7, axis=0)  # days
# y_naive = y_naive.loc[y_naive.index >= test_cutoff]
# naive_path = r'C:\Users\frede\PycharmProjects\Masters\results_app'
# pickle.dump(y_naive, open(os.path.join(naive_path, 'naive_forecast_all.pkl'), 'wb'))
#
# raise ValueError('stop')

#X_train = X.loc[X.index >= start_cutoff & X.index < val_cutoff]
X_train = X.loc[X.index < val_cutoff]
X_val = X.loc[(X.index >= val_cutoff) & (X.index < test_cutoff)]
X_test = X.loc[X.index >= test_cutoff]
#y_train = y.loc[y.index < val_cutoff & y.index >= start_cutoff]
y_train = y.loc[y.index < val_cutoff]
y_val = y.loc[(y.index >= val_cutoff) & (y.index < test_cutoff)]
y_test = y.loc[y.index >= test_cutoff]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)




class DNN4(nn.Module):
    def __init__(self, input_dim, output_dim, num_neurons, dropout_rate):
        # assure lenght of num_neurons is 3
        assert len(num_neurons) == 3

        super(DNN4, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, num_neurons[0])
        self.fc2 = nn.Linear(num_neurons[0], num_neurons[1])
        self.fc3 = nn.Linear(num_neurons[1], num_neurons[2])
        self.fc4 = nn.Linear(num_neurons[2], self.output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

# make dataloader
# and small test to see if it works, it does :)
# train_loader = DataLoader(dataset=list(zip(X_train.values, y_train.values)), batch_size=32, shuffle=False)
# val_loader = DataLoader(dataset=list(zip(X_val.values, y_val.values)), batch_size=32, shuffle=False)
#X.head()

# Tree structured parzen estimator

optimize_hyperparameters = False
batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512]
if optimize_hyperparameters:
    from hyperopt import fmin, tpe, hp, Trials

    epochs = 50
    criterion = nn.MSELoss() # maybe not this one, but for now

    # Define the search space
    space = {
        'weight_decay': hp.loguniform('weight_decay', -10, -1),
        'batch_size': hp.choice('batch_size', batch_sizes),
        'learning_rate': hp.loguniform('learning_rate', -8, -1),
        'num_neurons_1': hp.quniform('num_neurons_1', 32, 512, 32),
        'num_neurons_2': hp.quniform('num_neurons_2', 32, 512, 32),
        'num_neurons_3': hp.quniform('num_neurons_3', 32, 512, 32),
        'dropout_rate': hp.uniform('dropout_rate', 0, 0.5)
    }

    # Define the objective function
    def objective(params, input_dim=X_train.shape[1], output_dim=24):
        # Train and evaluate your model with the given hyperparameters
        # Return the validation accuracy or other metric you want to optimize
        params['num_neurons_1'] = int(params['num_neurons_1'])
        params['num_neurons_2'] = int(params['num_neurons_2'])
        params['num_neurons_3'] = int(params['num_neurons_3'])
        model = DNN4(input_dim, output_dim, [params['num_neurons_1'], params['num_neurons_2'], params['num_neurons_3']], params['dropout_rate']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
        # print layers
        # __loader__
        # fit scaler first, then transform: Fitting on training data then transforming on training and validation data

        XScaler = Scaler()
        # fit scaler without last 7 features
        XScaler.fit(X_train.iloc[:, :-7])


        X_train_scaled = XScaler.transform(X_train.iloc[:, :-7])
        # add dummies
        X_train_scaled = pd.concat([X_train_scaled, X_train.iloc[:, -7:]], axis=1)
        X_val_scaled = XScaler.transform(X_val.iloc[:, :-7])
        X_val_scaled = pd.concat([X_val_scaled, X_val.iloc[:, -7:]], axis=1)

        yScaler = Scaler()
        yScaler.fit(y_train)
        y_train_scaled = yScaler.transform(y_train)
        y_val_scaled = yScaler.transform(y_val)
        #train_dataset = Dataset()
        #train_dataset.load_data(X_train_scaled, y_train_scaled)
        train_loader = DataLoader(dataset=list(zip(X_train_scaled.values, y_train_scaled.values)), batch_size=params['batch_size'], shuffle=False)


        val_loader = DataLoader(dataset=list(zip(X_val_scaled.values, y_val_scaled.values)), batch_size=params['batch_size'], shuffle=False)
        train_losses = []
        val_losses = []
        best_val_loss = np.inf
        for epoch in range(epochs):
            train_loss = 0.0
            val_loss = 0.0
            model.train()
            for i, (inputs, labels) in enumerate(train_loader):
                # transfer to GPU
                inputs, labels = inputs.float().to(device), labels.float().to(device)


                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # print statistics
                train_loss += loss.item()

            model.eval()
            for i, (inputs, labels) in enumerate(val_loader):
                # transfer to GPU

                inputs, labels = inputs.float().to(device), labels.float().to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

            train_losses.append(train_loss/len(train_loader))
            val_losses.append(val_loss/len(val_loader))
            # early stopping
            if epoch % 5 == 0:

                # if val hasn't decreased for 5 epochs, stop
                if min(val_losses[-3:]) > best_val_loss:
                    break
                best_val_loss = min(val_losses[-3:])

        #accuracy = val_losses[-1]
        # min of last 3 val losses
        accuracy = min(val_losses[-3:])

        return accuracy

    # Define the TPE algorithm
    tpe_algorithm = tpe.suggest

    # Define the number of iterations
    max_evals = 1000

    # Initialize the trials object
    trials = Trials()

    # Run the TPE algorithm to optimize the hyperparameters
    best_params = fmin(objective, space, algo=tpe_algorithm, max_evals=max_evals, trials=trials, verbose=True)
    best_params['batch_size'] = batch_sizes[best_params['batch_size']]
    # Print the best hyperparameters
    print("Best hyperparameters:", best_params)
    # save best parameters to json
    param_path = r'C:\Users\frede\PycharmProjects\Masters\models\DNN\best_params.pkl'
    pickle.dump(best_params, open(param_path, 'wb'))
    trials_path = r'C:\Users\frede\PycharmProjects\Masters\models\DNN\trials.pkl'
    pickle.dump(trials, open(trials_path, 'wb'))


# load best hyper parameters
param_path = os.path.join('.', 'best_params.pkl')
best_params = pickle.load(open(param_path, 'rb'))
print(best_params)

raise Exception('stop here')
# get another param to see if it works
logger.info(f'Training model, with best hyperparameters {best_params}')


def build_train_model(model, date_to_forecast, X_training, y_training, X_validation, y_validation, batch_size, epochs, device_name='cpu'):
    # setup model

    best_val_loss = np.inf
    train_losses = []
    val_losses = []

    train_loader = DataLoader(dataset=list(zip(X_training.values, y_training.values)), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset=list(zip(X_validation.values, y_validation.values)), batch_size=batch_size, shuffle=False)

    model.train()
    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            # transfer to GPU
            inputs, labels = inputs.float().to(device_name), labels.float().to(device_name)
            assert not torch.isnan(inputs).any()
            assert not torch.isnan(labels).any()

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            train_loss += loss.item()



        model.eval()
        for i, (inputs, labels) in enumerate(val_loader):
            # transfer to GPU
            inputs, labels = inputs.float().to(device_name), labels.float().to(device_name)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

        train_losses.append(train_loss/len(train_loader))
        val_losses.append(val_loss/len(val_loader))
        # early stopping, every 5th epoch check if val loss has decreased
        if epoch % 5 == 0 and epoch > 1:

            # if val hasn't decreased for 5 epochs, stop
            if val_losses[-1] > best_val_loss:
                print(f'Terminated epoch {epoch + 1}/{epochs} early: train loss: {round(train_losses[-1], 2)}, val loss: {round(val_losses[-1], 2)}')
                break
            best_val_loss = val_losses[-1]

        logger.info(f'Epoch {epoch + 1}/{epochs} complete, train loss: {train_losses[-1]}, val loss: {val_losses[-1]}')

    #accuracy = val_losses[-1]
    # min of last 3 val losses
    # save model
    model_path = os.path.join(os.getcwd(), 'models', f'{date_to_forecast.strftime("%Y-%m-%d")}_DNN4_model.pth')
    torch.save(model.state_dict(), model_path)

    # save train and val losses
    losses_path = os.path.join(os.getcwd(), 'losses', f'{date_to_forecast.strftime("%Y-%m-%d")}_DNN4_losses.pkl')
    pickle.dump([train_losses, val_losses], open(losses_path, 'wb'))

    return model, train_losses, val_losses


# we want to go throught each day, train on all previous data except 1 week, then predict 1 day, then move on to next day
# we want to save the predictions for each day, and then evaluate the model on the whole dataset

# setup model
input_dim = X_train.shape[1]
output_dim = 24

best_params['num_neurons_1'] = int(best_params['num_neurons_1'])
best_params['num_neurons_2'] = int(best_params['num_neurons_2'])
best_params['num_neurons_3'] = int(best_params['num_neurons_3'])
best_params['batch_size'] = int(best_params['batch_size'])

model = DNN4(input_dim,
             output_dim,
             [best_params['num_neurons_1'], best_params['num_neurons_2'], best_params['num_neurons_3']],
             best_params['dropout_rate']).to(device)

optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
# loss
criterion = nn.MSELoss()

epochs = 50


# first we train model on all data except last week before testing, then recalibrate afterwards only on last year
XScaler = Scaler()
XScaler.fit(X_train.iloc[:, :-7])

X_train_scaled = XScaler.transform(X_train.iloc[:, :-7])
# add dummies
X_train_scaled = pd.concat([X_train_scaled, X_train.iloc[:, -7:]], axis=1)
X_val_scaled = XScaler.transform(X_val.iloc[:, :-7])
X_val_scaled = pd.concat([X_val_scaled, X_val.iloc[:, -7:]], axis=1)

yScaler = Scaler()
yScaler.fit(y_train)
y_train_scaled = yScaler.transform(y_train)
y_val_scaled = yScaler.transform(y_val)

model, train_losses_initial, val_losses_initial = build_train_model(model=model, date_to_forecast=test_cutoff, X_training=X_train_scaled, y_training=y_train_scaled, X_validation=X_val_scaled, y_validation=y_val_scaled, batch_size=best_params['batch_size'], epochs=epochs, device_name=device)
# save losses
losses_path = os.path.join(os.getcwd(), 'losses', f'DNN_initial_train_data_losses.pkl')
pickle.dump([train_losses_initial, val_losses_initial], open(losses_path, 'wb'))

predictions_path = os.path.join(os.getcwd(), 'predictions', f'DNN4_predictions.pkl')
predictions = []

calibration_window = pd.Timedelta(days=2 * 365)

time_start = time.time()
print(X_train.shape)
for i, date in enumerate(X_test.index):
    start_time = time.time()
    X_train_date = X[(X.index < date - pd.Timedelta(days=7)) & (X.index > date - calibration_window)]
    y_train_date = y[(y.index < date - pd.Timedelta(days=7)) & (X.index > date - calibration_window)]
    X_val_date = X[(X.index >= date - pd.Timedelta(days=7)) & (X.index < date)]
    y_val_date = y[(y.index >= date - pd.Timedelta(days=7)) & (y.index < date)]
    X_test_date = X[X.index == date]
    y_test_date = y[y.index == date]
    # Transform data
    XScaler = Scaler()
    XScaler.fit(X_train_date.iloc[:, :-7])
    X_train_date_transformed = XScaler.transform(X_train_date.iloc[:, :-7])
    # add dummies
    X_train_date_transformed = pd.concat([X_train_date_transformed, X_train_date.iloc[:, -7:]], axis=1)

    X_val_date_transformed = XScaler.transform(X_val_date.iloc[:, :-7])
    X_val_date_transformed = pd.concat([X_val_date_transformed, X_val_date.iloc[:, -7:]], axis=1)
    X_test_date_transformed = XScaler.transform(X_test_date.iloc[:, :-7])
    X_test_date_transformed = pd.concat([X_test_date_transformed, X_test_date.iloc[:, -7:]], axis=1)

    # Make test pandas dataframe into tensor
    X_test_date = torch.tensor(X_test_date.values).float().to(device)
    yScaler = Scaler()
    yScaler.fit(y_train_date)
    y_train_date = yScaler.transform(y_train_date)
    y_val_date = yScaler.transform(y_val_date)
    y_test_date = yScaler.transform(y_test_date)
    # train model
    model.train()
    model, train_losses, val_losses = build_train_model(model=model,
                                                        date_to_forecast=date,
                                                        X_training=X_train_date_transformed,
                                                        y_training=y_train_date,
                                                        X_validation=X_val_date_transformed,
                                                        y_validation=y_val_date,
                                                        batch_size=best_params['batch_size'],
                                                        epochs=epochs,
                                                        device_name=device)

    # predict and detach to cpu and make into pandas series
    X_test_date_transformed = torch.tensor(X_test_date_transformed.values).float().to(device)
    # disable dropout
    model = model.eval()
    y_pred = model(X_test_date_transformed).detach().cpu().numpy()
    y_pred = pd.DataFrame(y_pred, index=y_test_date.index)
    y_pred = yScaler.inverse_transform(y_pred)
    y_true = yScaler.inverse_transform(y_test_date)

    predictions.append(y_pred)

    logger.info(f'Training model for date {date} ({i+1}/{len(X_test.index)}): MAE: {mean_absolute_error(y_true, y_pred)}')
    print(date, 'MAE:', round(mean_absolute_error(y_true, y_pred), 2), 'time:', round(time.time() - start_time, 2))
    #print('predicted mean', round(y_pred.mean().mean(), 2))
    #print('true mean', round(y_true.mean().mean(), 2))
    time_elapsed = time.time() - time_start
    time_per_day = time_elapsed / (i + 1)
    days_remaining = len(X_test.index) - (i + 1)
    time_remaining = time_per_day * days_remaining
    print(f'Time remaining: {time_remaining / 3600} hours')

    # every quarter, save model and losses and predictions
    if date.month % 3 == 0 and date.day == 1:


        # save predictions
        predictions_path = os.path.join(os.getcwd(), 'predictions', f'{date.strftime("%Y-%m-%d")}_DNN4_predictions_2.pkl')
        pickle.dump(pd.concat(predictions, axis=0), open(predictions_path, 'wb'))



# save predictions
# concat predictions
predictions = pd.concat(predictions, axis=0)
predictions_path = os.path.join(os.getcwd(), 'predictions', f'DNN4_predictions_all_2.pkl')
pickle.dump(predictions, open(predictions_path, 'wb'))


# save actual values
actuals_path = os.path.join(os.getcwd(), 'predictions', f'DNN4_actuals_all.pkl')
pickle.dump(y_test, open(actuals_path, 'wb'))

# also save to app


# read predictions

os.chdir('..')
os.chdir('..')
os.chdir('results_app')
predictions_path = os.path.join(os.getcwd(), f'dnn4_preds_all.pkl')
pickle.dump(predictions, open(predictions_path, 'wb'))
