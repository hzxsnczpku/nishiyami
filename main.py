import os

used_gpu = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = used_gpu

import pandas as pd
from train import cross_validation
from models import *


def SVM_experiment(pca=None, img_shape=None, mode='rbf'):
    X_train, X_train_angle, Y_train = get_data("./input/train.json", train=True, img_shape=img_shape)
    X_test, X_test_angle, id = get_data("./input/test.json", train=False, img_shape=img_shape)
    model = SVM_Model(pca=pca, mode=mode)
    preds = cross_validation(model, X_train, X_train_angle, Y_train, X_test, X_test_angle, K=5)

    submission = pd.DataFrame()
    submission['id'] = id
    submission['is_iceberg'] = preds
    submission.to_csv('sub.csv', index=False)


def NN_experiment(model_kind, img_shape=None, lee=None, mixup=None):
    X_train, X_train_angle, Y_train = get_data("./input/train.json", train=True, img_shape=img_shape, lee=lee)
    X_test, X_test_angle, id = get_data("./input/test.json", train=False, img_shape=img_shape, lee=lee)

    model = Nerual_Network_Model(X_train.shape[1:], model_kind, batch_size=64, patience=100, mixup=mixup)
    preds = cross_validation(model, X_train, X_train_angle, Y_train, X_test, X_test_angle, K=5)

    submission = pd.DataFrame()
    submission['id'] = id
    submission['is_iceberg'] = preds
    submission.to_csv('sub.csv', index=False)


if __name__ == '__main__':
    SVM_experiment(img_shape=(32, 32), mode='rbf')
    SVM_experiment(img_shape=(32, 32), mode='linear')

    SVM_experiment(pca=20, mode='rbf')
    SVM_experiment(pca=20, mode='linear')
    SVM_experiment(pca=100, mode='rbf')
    SVM_experiment(pca=100, mode='linear')
    SVM_experiment(pca=200, mode='rbf')
    SVM_experiment(pca=200, mode='linear')

    NN_experiment('LeNet')
    NN_experiment('VGG16')
    NN_experiment('VGG19')
    NN_experiment('ResNet50')
    NN_experiment('DenseNet')
    NN_experiment('SE_ResNet')

    NN_experiment('LeNet', lee={'window': 8, 'var_noise': 4e-6})
    NN_experiment('VGG16', lee={'window': 8, 'var_noise': 4e-6})
    NN_experiment('VGG19', lee={'window': 8, 'var_noise': 4e-6})
    NN_experiment('ResNet50', lee={'window': 8, 'var_noise': 4e-6})

    NN_experiment('LeNet', mixup=0.5)
    NN_experiment('VGG16', mixup=0.5)
    NN_experiment('VGG19', mixup=0.5)
    NN_experiment('ResNet50', mixup=0.5)

