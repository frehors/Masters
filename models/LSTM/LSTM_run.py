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

def create_sequences(X, y, sequence_length):
    X_sequences = []
    y_sequences = []
    for i in range(len(X) - sequence_length):
        X_sequences.append(X[i:i+sequence_length])
        y_sequences.append(y[i+sequence_length])
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    X_sequences = torch.tensor(X_sequences).float()
    y_sequences = torch.tensor(y_sequences).float()
    return X_sequences, y_sequences

# Tree structured parzen estimator
optimize_hyperparameters = False

batch_size = [2, 4, 8, 16, 32, 64, 128, 256]
num_layers = [1, 2, 3, 4, 5]
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
        #train_loader = DataLoader(dataset=list(zip(X_train_scaled.values, y_train_scaled.values)), batch_size=params['batch_size'], shuffle=False)
        # make torch dataset
        X_train_sequences, y_train_sequences = create_sequences(X_train_scaled.values, y_train_scaled.values, params['sequence_length'])
        X_val_sequences, y_val_sequences = create_sequences(X_val_scaled.values, y_val_scaled.values, params['sequence_length'])
        train_dataset = TensorDataset(X_train_sequences, y_train_sequences)
        val_dataset = TensorDataset(X_val_sequences, y_val_sequences)
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)




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


best_params['batch_size'] = int(best_params['batch_size'])
best_params['sequence_length'] = int(best_params['sequence_length'])
best_params['hidden_size'] = int(best_params['hidden_size'])
best_params['num_layers'] = int(best_params['num_layers'])


model = LSTM(input_dim, best_params['hidden_size'], best_params['num_layers'], output_dim, best_params['dropout'], device_name='cpu')

optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
# loss
criterion = nn.MSELoss()




# first we train model on all data except last week before testing, then recalibrate afterwards only on 2 last years
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

model, _, _ = build_train_model(model=model, date_to_forecast=test_cutoff, X_training=X_train_scaled, y_training=y_train_scaled, X_validation=X_val_scaled, y_validation=y_val_scaled, batch_size=best_params['batch_size'], epochs=epochs, device_name=device)



predictions_path = os.path.join(os.getcwd(), 'predictions', f'DNN4_predictions.pkl')
predictions = []

epochs = 50


for i, date in enumerate(X_test.index):
    X_train_date = X[(X.index < date - pd.Timedelta(days=7)) & (X.index > date - pd.Timedelta(days=2 * 365))]
    y_train_date = y[(y.index < date - pd.Timedelta(days=7)) & (X.index > date - pd.Timedelta(days=2 * 365))]
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
    print(date, 'MAE:', round(mean_absolute_error(y_true, y_pred), 2))
    print('predicted mean', round(y_pred.mean().mean(), 2))
    print('true mean', round(y_true.mean().mean(), 2))

    # every quarter, save model and losses and predictions
    if date.month % 3 == 0 and date.day == 1:


        # save predictions
        predictions_path = os.path.join(os.getcwd(), 'predictions', f'{date.strftime("%Y-%m-%d")}_DNN4_predictions.pkl')
        pickle.dump(pd.concat(predictions, axis=0), open(predictions_path, 'wb'))



# save predictions
# concat predictions
predictions = pd.concat(predictions, axis=0)
predictions_path = os.path.join(os.getcwd(), 'predictions', f'DNN4_predictions_all.pkl')
pickle.dump(predictions, open(predictions_path, 'wb'))


# save actual values
actuals_path = os.path.join(os.getcwd(), 'predictions', f'DNN4_actuals_all.pkl')
pickle.dump(y_test, open(actuals_path, 'wb'))

