import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from data_loader import get_data, generator


def cross_validation(model, X_train, X_train_angle, Y_train, X_test, X_test_angle, K):
    folds = list(StratifiedKFold(n_splits=K, shuffle=True).split(X_train, Y_train))
    y_test_pred = 0
    y_train_pred = 0
    y_valid_pred = 0.0 * Y_train
    for j, (train_idx, test_idx) in enumerate(folds):
        print('\n===================FOLD=', j)
        X_cv = X_train[train_idx]
        Y_cv = Y_train[train_idx]
        X_angle_cv = X_train_angle[train_idx]
        X_hold = X_train[test_idx]
        Y_hold = Y_train[test_idx]
        X_angle_hold = X_train_angle[test_idx]

        # define file path and get callbacks
        model.reset()
        file_path = model.model_kind + "_model_weights_%s.hdf5" % j
        model.train(X_cv, X_angle_cv, Y_cv, X_hold, X_angle_hold, Y_hold, file_path)

        # Getting Training Score
        loss, acc = model.eval(X_cv, X_angle_cv, Y_cv)
        print('Train loss:', loss, ' Train accuracy:', acc)

        # Getting Test Score
        loss, acc = model.eval(X_hold, X_angle_hold, Y_hold)
        print('Test loss:', loss, ' Test accuracy:', acc)

        # Getting validation Score.
        y_valid_pred[test_idx] = model.predict(X_hold, X_angle_hold)
        y_test_pred += model.predict(X_test, X_test_angle)
        y_train_pred += model.predict(X_train, X_train_angle)

    y_test_pred = y_test_pred / K
    y_train_pred = y_train_pred / K

    print('\n Train Loss = ', log_loss(Y_train, y_train_pred))
    print(' Train Acc = ', np.sum(Y_train == (y_train_pred > 0.5)) / Y_train.shape[0])
    print(' Val Loss = ', log_loss(Y_train, y_valid_pred))
    print(' Val Acc  =', np.sum(Y_train == (y_valid_pred > 0.5)) / Y_train.shape[0])

    return y_test_pred
