from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def LeNet(input_shape):
    model = Sequential()
    model.add(
        Conv2D(6, (5, 5), padding='valid', activation='relu', kernel_initializer='he_normal', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(16, (5, 5), padding='valid', activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(84, activation='relu', kernel_initializer='he_normal'))

    return model