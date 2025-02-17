from tsai.all import *
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import optuna
from tsai.all import OmniScaleCNN

def objective(trial):
    # Hyperparameters to tune
    dsid = 'NATOPS'
    X_train, y_train, X_test, y_test = get_UCR_data(dsid, return_split=True)
    unique_labels = np.unique(y_train)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}

    y_train = np.array([label_map[label] for label in y_train])
    y_test = np.array([label_map[label] for label in y_test])

    bs = trial.suggest_categorical('batch_size', [32, 64, 128])
    lr = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    epochs = trial.suggest_int('epochs', 10, 50)

    tfms = [None, [Categorize()]]
    valid_size = 0.2
    if len(np.unique(y_train)) > len(y_train) * valid_size:
        valid_size = len(np.unique(y_train)) / len(y_train) + 0.1
    splits = get_splits(y_train, valid_size=valid_size)
    dsets = TSDatasets(X_train, y_train, tfms=tfms, splits=splits, inplace=True)
    dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=bs, batch_tfms=[TSStandardize()], num_workers=0)

    model = create_model(OmniScaleCNN, dls=dls)
    learn = Learner(dls, model, metrics=accuracy)

    try:
        learn.fit_one_cycle(epochs, lr)
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('inf')  

    val_loss, val_accuracy = learn.validate()

    return val_accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print('Best hyperparameters: ', study.best_params)
print('Best accuracy: ', study.best_value)
