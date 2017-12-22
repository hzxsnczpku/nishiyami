from nn_model.vgg import *
from nn_model.resnet import *
from nn_model.densenet import *
from nn_model.se_resnet import *
from nn_model.lenet import *
from keras.optimizers import Adam


def get_combined_Model(img_shape, model, combined=True):
    base_model = model(input_shape=img_shape)
    x = base_model.output
    input_2 = Input(shape=[1], name="angle")

    if combined:
        angle_layer = Dense(1, )(input_2)
        merge_one = concatenate([x, angle_layer])
        merge_one = Dense(512, activation='relu', name='fc2')(merge_one)
        merge_one = Dropout(0.3)(merge_one)
        merge_one = Dense(512, activation='relu', name='fc3')(merge_one)
        merge_one = Dropout(0.3)(merge_one)
        predictions = Dense(1, activation='sigmoid')(merge_one)
    else:
        predictions = Dense(1, activation='sigmoid')(x)

    model = Model(input=[base_model.input, input_2], output=predictions)
    # sgd = Adam(lr=1e-3)
    sgd = SGD(lr=1e-3, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()

    return model


models = {'LeNet': LeNet,
          'VGG16': VGG16,
          'VGG19': VGG19,
          'ResNet50': ResNet50,
          'DenseNet121': densenet121,
          'SE_ResNet50': SENet50}
