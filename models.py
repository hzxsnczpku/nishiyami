from nn_model import *
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from data_loader import get_data, generator
from sklearn import svm
import numpy as np


class Nerual_Network_Model:
    def __init__(self, input_shape, model_kind, batch_size=64, patience=10, combined=True, mixup=None):
        self.input_shape = input_shape
        self.model_kind = model_kind
        self.batch_size = batch_size
        self.patience = patience
        self.mixup = mixup
        self.model = get_combined_Model(self.input_shape, models[self.model_kind], combined=combined)

    def reset(self):
        self.model = get_combined_Model(self.input_shape, models[self.model_kind])

    def train(self, X_train, X_train_angle, Y_train, X_val, X_val_angle, Y_val, file_path):
        callbacks = [EarlyStopping('val_loss', patience=self.patience, mode="min"),
                     ModelCheckpoint(file_path, save_best_only=True)]
        gen_flow = generator(X_train, X_train_angle, Y_train, batch_size=self.batch_size, mixup=self.mixup)
        history = self.model.fit_generator(gen_flow, steps_per_epoch=24, epochs=1000, shuffle=True, verbose=1,
                            validation_data=([X_val, X_val_angle], Y_val), callbacks=callbacks)
        self.model.load_weights(filepath=file_path)

        return history

    def eval(self, X, X_angle, Y):
        score = self.model.evaluate([X, X_angle], Y, verbose=0)
        return score[0], score[1]

    def predict(self, X, X_angle):
        temp = self.model.predict([X, X_angle])
        return temp.reshape(temp.shape[0])


class SVM_Model:
    def __init__(self, pca=None, mode='rbf'):
        self.model = svm.SVC(C=1.0, kernel=mode, decision_function_shape='ovr', probability=True)
        self.pca = pca

    def reset(self):
        pass

    def train(self, X_train, X_train_angle, Y_train, X_val, X_val_angle, Y_val, file_path):
        if self.pca is not None:
            im1 = X_train[:, :, :, 0]
            im2 = X_train[:, :, :, 1]
            im1 = np.reshape(im1, (im1.shape[0], -1))
            im2 = np.reshape(im1, (im2.shape[0], -1))
            self.im1_mean = np.mean(im1, axis=0)
            self.im1_std = np.std(im1, axis=0)
            self.im2_mean = np.mean(im2, axis=0)
            self.im2_std = np.std(im2, axis=0)
            U1, self.s1, self.V1 = np.linalg.svd((im1 - self.im1_mean) / self.im1_std, full_matrices=0)
            U2, self.s2, self.V2 = np.linalg.svd((im2 - self.im2_mean) / self.im2_std, full_matrices=0)
            X_train = np.hstack((U1[:, :self.pca], U2[:, :self.pca]))

        X_train_reshape = X_train.reshape((X_train.shape[0], -1))
        self.model.fit(X_train_reshape, Y_train)

    def eval(self, X, X_angle, Y):
        if self.pca is not None:
            im1 = X[:, :, :, 0]
            im2 = X[:, :, :, 1]
            im1 = np.reshape(im1, (im1.shape[0], -1))
            im2 = np.reshape(im1, (im2.shape[0], -1))
            im1 = (im1 - self.im1_mean) / self.im1_std
            im2 = (im2 - self.im2_mean) / self.im2_std
            U1 = np.dot(np.dot(im1, self.V1.T), np.diag(1 / self.s1))
            U2 = np.dot(np.dot(im2, self.V2.T), np.diag(1 / self.s2))
            X = np.hstack((U1[:, :self.pca], U2[:, :self.pca]))
        X_reshape = X.reshape((X.shape[0], -1))
        Y = np.array([y for y in Y])

        pred = self.model.predict_proba(X_reshape)[:, 1]
        loss = np.mean(-Y * np.log(pred) - (1 - Y) * np.log(1 - pred))
        acc = np.sum((pred > 0.5) == Y) / np.shape(Y)[0]

        return loss, acc

    def predict(self, X, X_angle):
        if self.pca is not None:
            im1 = X[:, :, :, 0]
            im2 = X[:, :, :, 1]
            im1 = np.reshape(im1, (im1.shape[0], -1))
            im2 = np.reshape(im1, (im2.shape[0], -1))
            im1 = (im1 - self.im1_mean) / self.im1_std
            im2 = (im2 - self.im2_mean) / self.im2_std
            U1 = np.dot(np.dot(im1, self.V1.T), np.diag(1 / self.s1))
            U2 = np.dot(np.dot(im2, self.V2.T), np.diag(1 / self.s2))
            X = np.hstack((U1[:, :self.pca], U2[:, :self.pca]))
        X_reshape = X.reshape((X.shape[0], -1))
        pred = self.model.predict_proba(X_reshape)[:, 1]

        return pred
