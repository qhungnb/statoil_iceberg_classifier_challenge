from keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout,
    Input, GlobalAvgPool2D, GlobalMaxPool2D, Flatten,
    BatchNormalization, Activation, LeakyReLU, AveragePooling2D
)
from keras.applications.resnet50 import (
    conv_block,
    identity_block,
    ResNet50
)
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model


def custom_conv_block(x, nf=8, k=3, s=1, nb=2, p_act='elu'):
    for i in range(nb):
        x = Conv2D(filters=nf, kernel_size=(k, k), strides=(s, s),
                   activation=p_act,
                   padding='same', kernel_initializer='he_uniform')(x)

    return x


def dense_block(x, h=32, d=0.5, m=0., p_act='elu'):
    return Dropout(d)(BatchNormalization(momentum=m)(Dense(h, activation=p_act)(x)))


def bn_pooling(x, k=2, s=2, m=0):
    return MaxPooling2D((k, k), strides=(s, s))(BatchNormalization(momentum=m)(x))


def get_cnn_model(img_shape=(75, 75, 3), num_classes=1, f=8, h=128):
    """
    This model structure is inspired and modified from the following kernel
    https://www.kaggle.com/knowledgegrappler/a-keras-prototype-0-21174-on-pl
    img_shape: dimension for input image
    f: filters of first conv blocks and generate filters in the following
       blocks acorrdingly
    h: units in dense hidden layer
    """

    # model
    bn_model = 0
    p_activation = 'elu'

    #
    input_img = Input(shape=img_shape, name='img_inputs')
    input_img_bn = BatchNormalization(momentum=bn_model)(input_img)
    #
    input_meta = Input(shape=[1], name='angle')
    input_meta_bn = BatchNormalization(momentum=bn_model)(input_meta)

    # img_1
    # img_1:block_1
    img_1 = custom_conv_block(input_img_bn, nf=f, k=3, s=1, nb=3, p_act=p_activation)
    img_1 = bn_pooling(img_1, k=3, s=3, m=0)

    # img_1:block_2
    f *= 2
    img_1 = Dropout(0.2)(img_1)
    img_1 = custom_conv_block(img_1, nf=f, k=3, s=1, nb=3, p_act=p_activation)
    img_1 = bn_pooling(img_1, k=3, s=2, m=0)

    # img_1:block_3
    f *= 2
    img_1 = Dropout(0.2)(img_1)
    img_1 = custom_conv_block(img_1, nf=f, k=3, s=1, nb=3, p_act=p_activation)
    img_1 = bn_pooling(img_1, k=3, s=3, m=0)

    # img_1:block_4
    f *= 2
    img_1 = Dropout(0.2)(img_1)
    img_1 = custom_conv_block(img_1, nf=f, k=3, s=1, nb=3, p_act=p_activation)
    img_1 = Dropout(0.2)(img_1)
    img_1 = BatchNormalization(momentum=bn_model)(GlobalMaxPooling2D()(img_1))

    # img 2
    img_2 = custom_conv_block(input_img_bn, nf=f, k=3, s=1, nb=6, p_act=p_activation)
    img_2 = Dropout(0.2)(img_2)
    img_2 = BatchNormalization(momentum=bn_model)(GlobalMaxPooling2D()(img_2))

    # full connect
    concat = (Concatenate()([img_1, img_2, input_meta_bn]))
    x = dense_block(concat, h=h)
    x = dense_block(x, h=h)
    output = Dense(num_classes, activation='sigmoid')(x)

    model = Model([input_img, input_meta], output)

    return model


def Resnet(input_shape=None, classes=1, bn_momentum=0):
    bn_axis = 3
    inc_angle_input = Input(shape=[1], name='inc_angle')
    img_input = Input(shape=input_shape, name='image_input')

    img_input_bn = BatchNormalization(axis=bn_axis, momentum=bn_momentum)(img_input)
    inc_angle_bn = BatchNormalization(momentum=bn_momentum)(inc_angle_input)

    image_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(img_input_bn)
    image_1 = BatchNormalization(axis=bn_axis, momentum=bn_momentum)(image_1)
    image_1 = Activation('relu')(image_1)
    image_1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(image_1)

    image_1 = conv_block(image_1, 3, [32, 32, 64], stage=2, block='a')
    image_1 = Dropout(0.2)(image_1)
    image_1 = identity_block(image_1, 3, [32, 32, 64], stage=2, block='b')
    image_1 = Dropout(0.2)(image_1)
    image_1 = identity_block(image_1, 3, [32, 32, 64], stage=2, block='c')

    image_1 = conv_block(image_1, 3, [32, 64, 128], stage=3, block='a')
    image_1 = Dropout(0.2)(image_1)
    image_1 = identity_block(image_1, 3, [32, 64, 128], stage=3, block='b')
    image_1 = Dropout(0.2)(image_1)
    image_1 = identity_block(image_1, 3, [32, 64, 128], stage=3, block='c')
    image_1 = Dropout(0.2)(image_1)
    image_1 = identity_block(image_1, 3, [32, 64, 128], stage=3, block='d')

    image_1 = conv_block(image_1, 3, [32, 32, 128], stage=5, block='a')
    image_1 = Dropout(0.2)(image_1)
    image_1 = identity_block(image_1, 3, [32, 32, 128], stage=5, block='b')
    image_1 = Dropout(0.2)(image_1)
    image_1 = identity_block(image_1, 3, [32, 32, 128], stage=5, block='c')

    image_1 = MaxPooling2D(pool_size=(5, 5), name='max_pool_1')(image_1)

    image_1 = Flatten()(image_1)

    image_2 = custom_conv_block(img_input_bn, nf=64, k=3, s=1, nb=6, p_act='relu')
    image_2 = Dropout(0.2)(image_2)
    image_2 = BatchNormalization(momentum=bn_momentum)(GlobalMaxPooling2D()(image_2))

    output = Concatenate(axis=-1)([image_1, image_2, inc_angle_bn])

    # output = Dense(units=128, activation='relu', name='fc_1')(output)
    # output = Dropout(0.5)(output)

    output = Dense(classes, activation='sigmoid', name='fc_last')(output)

    model = Model(inputs=[img_input, inc_angle_input], outputs=output, name='simple resnet')
    return model


if __name__ == '__main__':
    # model = get_cnn_model(f=8, h=64)

    # model = ResNet50(include_top=True, weights=None, input_shape=(200, 200, 3))
    model = Resnet(input_shape=(75, 75, 3), classes=1)

    print(model.summary())
