from keras.layers import BatchNormalization, Activation, Lambda, Conv2D, concatenate, GlobalAveragePooling2D, Dense, \
    Input, Reshape, add, multiply
from keras import Model
from keras.initializers import he_normal
from keras import regularizers
import math


def SENet50(input_shape):
    weight_decay = 0.0005
    cardinality = 4  # 4 or 8 or 16 or 32
    base_width = 64
    inplanes = 64
    expansion = 4

    def add_common_layer(x):
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def group_conv(x, planes, stride):
        h = planes // cardinality
        groups = []
        for i in range(cardinality):
            group = Lambda(lambda z: z[:, :, :, i * h: i * h + h])(x)
            groups.append(Conv2D(h, kernel_size=(3, 3), strides=stride, kernel_initializer=he_normal(),
                                 kernel_regularizer=regularizers.l2(weight_decay), padding='same', use_bias=False)(
                group))
        x = concatenate(groups)
        return x

    def residual_block(x, planes, stride=(1, 1)):
        D = int(math.floor(planes * (base_width / 64.0)))
        C = cardinality

        shortcut = x

        y = Conv2D(D * C, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer=he_normal(),
                   kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(shortcut)
        y = add_common_layer(y)

        y = group_conv(y, D * C, stride)
        y = add_common_layer(y)

        y = Conv2D(planes * expansion, kernel_size=(1, 1), strides=(1, 1), padding='same',
                   kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(y)
        y = add_common_layer(y)

        if stride != (1, 1) or inplanes != planes * expansion:
            shortcut = Conv2D(planes * expansion, kernel_size=(1, 1), strides=stride, padding='same',
                              kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(weight_decay),
                              use_bias=False)(x)
            shortcut = BatchNormalization()(shortcut)

        y = squeeze_excite_block(y, planes * expansion)
        y = add([y, shortcut])
        y = Activation('relu')(y)
        return y

    def residual_layer(x, blocks, planes, stride=(1, 1)):
        x = residual_block(x, planes, stride)
        nonlocal inplanes
        inplanes = planes * expansion
        for i in range(1, blocks):
            x = residual_block(x, planes)
        return x

    def squeeze_excite_block(input, filters, ratio=16):
        init = input
        se_shape = (1, 1, filters)
        se = GlobalAveragePooling2D()(init)
        se = Reshape(se_shape)(se)
        se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(se)
        x = multiply([init, se])
        return x

    def conv3x3(x, filters):
        x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=he_normal(),
                   kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(x)
        return add_common_layer(x)

    img_input = Input(shape=input_shape)

    x = conv3x3(img_input, 64)
    x = residual_layer(x, 3, 64)
    x = residual_layer(x, 3, 128, stride=(2, 2))
    x = residual_layer(x, 3, 256, stride=(2, 2))
    x = GlobalAveragePooling2D()(x)

    model = Model(img_input, x)

    return model