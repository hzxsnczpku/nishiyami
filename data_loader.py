import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import cv2
from utils import lee_filter


def get_data(filename, img_shape=None, train=True, three_dim=True, change_to_linear=False, lee=None, pca=None):
    data = pd.read_json(filename)
    data['inc_angle'] = pd.to_numeric(data['inc_angle'], errors='coerce')
    data['inc_angle'] = data['inc_angle'].fillna(method='pad')

    band_1 = [np.array(b).reshape(75, 75).astype(np.float32) for b in data["band_1"]]
    band_2 = [np.array(b).reshape(75, 75).astype(np.float32) for b in data["band_2"]]

    if lee is not None:
        band_1 = [lee_filter(b, lee['window'], lee['var_noise']) for b in band_1]
        band_2 = [lee_filter(b, lee['window'], lee['var_noise']) for b in band_2]
    if img_shape is not None:
        band_1 = [cv2.resize(np.array(b).reshape(75, 75), img_shape).astype(np.float32) for b in band_1]
        band_2 = [cv2.resize(np.array(b).reshape(75, 75), img_shape).astype(np.float32) for b in band_2]

    band_1 = np.array(band_1)
    band_2 = np.array(band_2)
    band_3 = (band_1 + band_2) / 2

    if three_dim:
        X_data = np.concatenate([band_1[:, :, :, np.newaxis], band_2[:, :, :, np.newaxis], band_3[:, :, :, np.newaxis]],
                                axis=-1)
    else:
        X_data = np.concatenate([band_1[:, :, :, np.newaxis], band_2[:, :, :, np.newaxis]], axis=-1)
    X_angle = data['inc_angle']

    if change_to_linear:
        X_data = np.power(10, X_data / 10)

    if train:
        target = data['is_iceberg']
        return X_data, X_angle, target
    else:
        return X_data, X_angle, data['id']


def generator(X1, X2, y, batch_size, mixup=None):
    gen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, zoom_range=0.2, rotation_range=10)
    genX1 = gen.flow(X1, y, batch_size=batch_size, seed=55)
    genX2 = gen.flow(X1, X2, batch_size=batch_size, seed=55)
    if mixup is not None:
        genX1_2 = gen.flow(X1, y, batch_size=batch_size, seed=90)
        genX2_2 = gen.flow(X1, X2, batch_size=batch_size, seed=90)
        while True:
            X1i = genX1.next()
            X2i = genX2.next()
            lam = np.random.beta(mixup, mixup)
            X1i_2 = genX1_2.next()
            X2i_2 = genX2_2.next()
            yield [lam * X1i[0] + (1 - lam) * X1i_2[0], lam * X2i[1] + (1 - lam) * X2i_2[1]], lam * X1i[1] + (1 - lam) * \
                  X1i_2[1]
    else:
        while True:
            X1i = genX1.next()
            X2i = genX2.next()
            yield [X1i[0], X2i[1]], X1i[1]
