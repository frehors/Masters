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
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import hyperopt
import pickle
import time

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    , handlers=[logging.FileHandler('LSTM_run.log'), logging.StreamHandler()])
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


class Scaler:

    def __init__(self, median=None, mad=None):
        self.median = None
        self.mad = None

    def fit(self, data):
        if isinstance(data, pd.Series):
            data = pd.DataFrame(data)
        self.median = data.median(axis=0).to_numpy().reshape(1, len(data.columns))
        # calculate median absolute deviation
        self.mad = data.sub(data.median(axis=0), axis=1).abs().median(axis=0).to_numpy().reshape(1, len(data.columns))
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

# now y
# make y to dataframe first, should alreadt be, but just to be sure
y = pd.DataFrame(y)
y = y.pivot_table(index=y.index.date, columns=y.index.hour, values=y.columns)
# join multiindex columns to one, price with hour number
y.index = pd.to_datetime(y.index)

y = y.dropna()


val_cutoff = pd.to_datetime('2020-07-01')
test_cutoff = pd.to_datetime('2021-01-01')
X_train = X.loc[X.index < val_cutoff]
X_val = X.loc[(X.index >= val_cutoff) & (X.index < test_cutoff)]
X_test = X.loc[X.index >= test_cutoff]
y_train = y.loc[y.index < val_cutoff]
y_val = y.loc[(y.index >= val_cutoff) & (y.index < test_cutoff)]
y_test = y.loc[y.index >= test_cutoff]

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_().to(device)
        out, (hn, cn) = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])
        return out

def create_sequences(X, sequence_length):
    X_sequences = []
    for i in range(len(X) - sequence_length):
        X_sequences.append(X[i:i+sequence_length])
    X_sequences = np.array(X_sequences)
    X_sequences = torch.tensor(X_sequences).float()
    # swap dim 1 and 2


    return X_sequences


def train_val_test_sequences(train_date_from, val_cutoff, test_cutoff, seq_length, batch_size):
    # Scale data according to training data
    # sequence for all data.
    # output only index between data_date_from and data_date_to
    # The dates are inclusive
    func_time = time.time()
    XScaler = Scaler()
    X_local = X.copy()
    y_local = y.copy()
    X_train = X_local[(X_local.index >= train_date_from) & (X_local.index < val_cutoff)]
    XScaler.fit(X_train.iloc[:, :-7])
    X_scaled = XScaler.transform(X_local.iloc[:, :-7])
    # add dummies
    X_scaled = pd.concat([X_scaled, X_local.iloc[:, -7:]], axis=1)
    yScaler = Scaler()
    yScaler.fit(y_local[(y_local.index >= train_date_from) & (y_local.index < val_cutoff)])
    y_scaled = yScaler.transform(y)
    # create sequences for all data
    X_seq = create_sequences(X_scaled.values, seq_length)
    # get index of data to use

    # we need to drop sequence length from the index as we have lost that many rows
    X_local = X_local.iloc[seq_length:]
    # if test is iterable do smth
    val_idx = np.where(X_local.index == val_cutoff)[0][0]
    test_idx = np.where(X_local.index == test_cutoff)[0][0]

    # split X_seq and y_seq tensors into train, val and test sets
    X_train_seq = X_seq[:val_idx]
    y_train_seq = y_scaled.iloc[:val_idx] # slice is not including end, works like range
    X_val_seq = X_seq[val_idx:test_idx]
    y_val_seq = y_scaled.iloc[val_idx:test_idx]
    X_test_seq = X_seq[test_idx] # gets the index of the test date
    y_test_seq = y_scaled.iloc[test_idx]

    # y to tensor
    y_train_seq = torch.tensor(y_train_seq.values).float()
    y_val_seq = torch.tensor(y_val_seq.values).float()
    y_test_seq = torch.tensor(y_test_seq.values).float()

    # add dim for test as this is only one dim, then we now both have row
    X_test_seq = X_test_seq.unsqueeze(0)
    y_test_seq = y_test_seq.unsqueeze(0)



    train_loader = DataLoader(TensorDataset(X_train_seq, y_train_seq), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(X_val_seq, y_val_seq), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_seq, y_test_seq), batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader, XScaler, yScaler

# Tree structured parzen estimator

optimize_hyperparameters = False

batch_size = [2, 4, 8, 16, 32, 64, 128]
num_layers = [1, 2, 3, 4]
seq_length = [1, 2, 3, 4, 6, 8, 12, 24]
if optimize_hyperparameters:
    from hyperopt import fmin, tpe, hp, Trials

    epochs = 50
    criterion = nn.MSELoss() # maybe not this one, but for now

    # Define the search space
    space = {
        'weight_decay': hp.loguniform('weight_decay', -10, -1),
        'num_layers': hp.choice('num_layers', num_layers),
        'sequence_length': hp.choice('sequence_length', seq_length), # 1-24 hours
        'batch_size': hp.choice('batch_size', batch_size),
        'learning_rate': hp.loguniform('learning_rate', -8, -1),
        'hidden_size': hp.quniform('hidden_size', 32, 512, 32),
        'dropout_rate': hp.uniform('dropout_rate', 0, 0.5)
    }

    # Define the objective function
    def objective(params, input_dim=X_train.shape[1], output_dim=24):
        # Train and evaluate your model with the given hyperparameters
        # Return the validation accuracy or other metric you want to optimize

        params['batch_size'] = int(params['batch_size'])
        params['hidden_size'] = int(params['hidden_size'])
        params['num_layers'] = int(params['num_layers'])
        params['sequence_length'] = int(params['sequence_length'])

        if params['num_layers'] == 1:  # if only one layer, no dropout as this occurs between layers
            params['dropout_rate'] = 0

        model = LSTM(input_size=input_dim, hidden_size=params['hidden_size'], num_layers=params['num_layers'], output_size=output_dim, dropout=params['dropout_rate']).to(device)

        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])

        # fit scaler first, then transform: Fitting on training data then transforming on training and validation data
        # sequence length after start date in X index
        # get index value of 'sequence_length' days after start date
        train_date_from_hyper = X_train.index[params['sequence_length']]
        # get first index value of validation data and test data
        val_date_from_hyper = X_val.index[0]
        test_date_from_hyper = X_test.index[0]

        train_loader, val_loader, test_loader, XScaler, yScaler = train_val_test_sequences(train_date_from=train_date_from_hyper,
                                                                                           val_cutoff=val_date_from_hyper,
                                                                                           test_cutoff=test_date_from_hyper,
                                                                                           seq_length=params['sequence_length'],
                                                                                           batch_size=params['batch_size'])



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

    # Print the best hyperparameters
    print("Best hyperparameters:", best_params)
    # save best parameters to pkl
    param_path = r'C:\Users\frede\PycharmProjects\Masters\models\LSTM\best_params.pkl'
    pickle.dump(best_params, open(param_path, 'wb'))
    trials_path = r'C:\Users\frede\PycharmProjects\Masters\models\LSTM\trials.pkl'
    pickle.dump(trials, open(trials_path, 'wb'))


# load best hyper parameters
param_path = os.path.join('.', 'best_params.pkl')
best_params = pickle.load(open(param_path, 'rb'))

# get another param to see if it works
logger.info(f'Training model, with best hyperparameters {best_params}')


def train_model(model, date_to_forecast, train_loader, val_loader, batch_size, epochs):
    # setup model

    best_val_loss = np.inf
    train_losses = []
    val_losses = []


    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0

        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            # transfer to GPU
            inputs, labels = inputs.float().to(device), labels.float().to(device)
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
            inputs, labels = inputs.float().to(device), labels.float().to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

        train_losses.append(train_loss/len(train_loader))
        val_losses.append(val_loss/len(val_loader))
        # early stopping, every 5th epoch check if val loss has decreased
        if epoch % 5 == 0:

            # if val hasn't decreased for 5 epochs, stop
            if val_losses[-1] > best_val_loss:
                print(f'Terminated epoch {epoch + 1}/{epochs} early: train loss: {round(train_losses[-1], 2)}, val loss: {round(val_losses[-1], 2)}')
                break
            best_val_loss = val_losses[-1]

        logger.info(f'Epoch {epoch + 1}/{epochs} complete, train loss: {train_losses[-1]}, val loss: {val_losses[-1]}')

    #accuracy = val_losses[-1]
    # min of last 3 val losses
    # save model
    model_path = os.path.join(os.getcwd(), 'models', f'{date_to_forecast.strftime("%Y-%m-%d")}_lstm_model.pth')
    torch.save(model.state_dict(), model_path)

    # save train and val losses
    losses_path = os.path.join(os.getcwd(), 'losses', f'{date_to_forecast.strftime("%Y-%m-%d")}_lstm_losses.pkl')
    pickle.dump([train_losses, val_losses], open(losses_path, 'wb'))

    return model, train_losses, val_losses


# we want to go throught each day, train on all previous data except 1 week, then predict 1 day, then move on to next day
# we want to save the predictions for each day, and then evaluate the model on the whole dataset

# setup model
input_dim = X_train.shape[1]
output_dim = 24

epochs = 50
##################################################
best_params['batch_size'] = int(batch_size[best_params['batch_size']])
best_params['hidden_size'] = int(best_params['hidden_size'])
best_params['num_layers'] = int(num_layers[best_params['num_layers']])
best_params['sequence_length'] = int(seq_length[best_params['sequence_length']])
if best_params['num_layers'] == 1:  # if only one layer, no dropout as this occurs between layers
    best_params['dropout_rate'] = 0

print(best_params)
########### SETUP MODEL AND OPTIMIZER WITH BEST HYPERPARAMETERS #############


## Train model initially on training data
train_loader_init, val_loader_init, _, _, _ = train_val_test_sequences(train_date_from=X.index[0],
                                                                                      val_cutoff=val_cutoff,
                                                                                      test_cutoff=test_cutoff,
                                                                                      seq_length=best_params['sequence_length'],
                                                                                      batch_size=best_params['batch_size'])

model = LSTM(input_size=input_dim, hidden_size=best_params['hidden_size'], num_layers=best_params['num_layers'], output_size=output_dim, dropout=best_params['dropout_rate']).to(device)

optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
criterion = nn.MSELoss()
model.train()
model, train_losses_initial, val_losses_initial = train_model(model=model,
                                                                    date_to_forecast=test_cutoff,
                                                                    train_loader=train_loader_init,
                                                                    val_loader=val_loader_init,
                                                                    batch_size=best_params['batch_size'],
                                                                    epochs=epochs)
#save losses
losses_path = os.path.join(os.getcwd(), 'losses', f'LSTM_initial_train_data_losses.pkl')
pickle.dump([train_losses_initial, val_losses_initial], open(losses_path, 'wb'))

predictions_path = os.path.join(os.getcwd(), 'predictions', f'LSTM_predictions.pkl')
predictions = []



calibration_window = pd.Timedelta(days=2 * 365)

# only keep last month of X_test - crashed during last part


# load newest model
# load_model = True
# if load_model:
#     model_path = r'C:\Users\frede\PycharmProjects\Masters\models\LSTM\models\2022-11-30_lstm_model.pth'
#     model = LSTM(input_size=input_dim, hidden_size=best_params['hidden_size'], num_layers=best_params['num_layers'], output_size=output_dim, dropout=best_params['dropout_rate']).to(device)
#     model.load_state_dict(torch.load(model_path))
#     optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
#     criterion = nn.MSELoss()

start_time = time.time()
for i, date in enumerate(X_test.index):
    train_date_from = date - calibration_window
    val_cutoff = date - pd.Timedelta(days=7)
    test_cutoff = date
    train_loader_date, val_loader_date, test_loader, X_scaler_date, y_scaler_date = train_val_test_sequences(train_date_from,
                                                                                                             val_cutoff,
                                                                                                             test_cutoff,
                                                                                                             best_params['sequence_length'],
                                                                                                             batch_size = best_params['batch_size'])
    # train model
    model.train()
    model, train_losses, val_losses = train_model(model=model,
                                                        date_to_forecast=date,
                                                        train_loader=train_loader_date,
                                                        val_loader=val_loader_date,
                                                        batch_size=best_params['batch_size'],
                                                        epochs=epochs)

    model = model.eval()
    for input, target in test_loader:
        input = input.to(device)
        target = target.to(device)
        pred = model(input).detach().cpu().numpy()

    # make prediction into dataframe 1 row, y columns and index of date
    y_pred = pd.DataFrame(pred, index=[date])
    y_pred = y_scaler_date.inverse_transform(y_pred)
    y_true = y_test[y_test.index == date]
    predictions.append(y_pred)
    # expectd runtime
    elapsed_time = time.time() - start_time
    expected_time = elapsed_time / (i + 1) * len(X_test.index)
    logger.info(f'Date: {date} Expected time remaining: {(expected_time - elapsed_time) / 3600:.2f} hours, MAE: {mean_absolute_error(y_pred, y_true):.2f}')
    print(f'Date: {date} Expected time remaining: {(expected_time - elapsed_time) / 3600:.2f} hours, MAE: {mean_absolute_error(y_pred, y_true):.2f}')
    # every 3 months save predictions
    if date.month % 3 == 0 and date.day == 30:
        # save predictions
        predictions_path = os.path.join(os.getcwd(), 'predictions', f'{date.strftime("%Y-%m-%d")}_lstm_predictions_1.pkl')
        pickle.dump(pd.concat(predictions, axis=0), open(predictions_path, 'wb'))



# # read all previous predictions
# prediction_files = ['2021-03-01_lstm_predictions.pkl',
#                     '2021-06-01_lstm_predictions.pkl',
#                     '2021-09-01_lstm_predictions.pkl',
#                     '2021-12-01_lstm_predictions.pkl',
#                     '2022-03-01_lstm_predictions.pkl',
#                     '2022-06-01_lstm_predictions.pkl',
#                     '2022-09-01_lstm_predictions.pkl',
#                     '2022-12-01_lstm_predictions.pkl']
#
# for file in os.listdir(os.path.join(os.getcwd(), 'predictions')):
#     if file in prediction_files:
#         predictions.append(pickle.load(open(os.path.join(os.getcwd(), 'predictions', file), 'rb')))
# # concat predictions

lstm_all_preds_path = os.path.join(os.getcwd(), 'predictions', f'lstm_preds_all_2.pkl')
predictions = pd.concat(predictions, axis=0)
pickle.dump(predictions, open(lstm_all_preds_path, 'wb'))


# save actual values
actuals_path = os.path.join(os.getcwd(), 'predictions', f'lstm_actuals_all.pkl')
pickle.dump(y_test, open(actuals_path, 'wb'))

# read predictions

os.chdir('..')
os.chdir('..')
os.chdir('results_app')
predictions_path = os.path.join(os.getcwd(), f'lstm_preds_all.pkl')
pickle.dump(predictions, open(predictions_path, 'wb'))

