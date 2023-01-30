import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses, Model


def inception(
    x,
    filters_1x1,
    filters_3x3_reduce,
    filters_3x3,
    filters_5x5_reduce,
    filters_5x5,
    filters_pool
):
    """Inception method for GoogLeNetV1

    :param x: _description_
    :type x: _type_
    :param filters_1x1: _description_
    :type filters_1x1: _type_
    :param filters_3x3_reduce: _description_
    :type filters_3x3_reduce: _type_
    :param filters_3x3: _description_
    :type filters_3x3: _type_
    :param filters_5x5_reduce: _description_
    :type filters_5x5_reduce: _type_
    :param filters_5x5: _description_
    :type filters_5x5: _type_
    :param filters_pool: _description_
    :type filters_pool: _type_
    :return: _description_
    :rtype: _type_
    """
    path1 = layers.Conv2D(
        filters_1x1,
        (1, 1),
        padding='same',
        activation='relu'
    )(x)

    path2 = layers.Conv2D(
        filters_3x3_reduce, (1, 1),
        padding='same', activation='relu')(x)
    path2 = layers.Conv2D(
        filters_3x3, (1, 1),
        padding='same', activation='relu')(path2)

    path3 = layers.Conv2D(
        filters_5x5_reduce, (1, 1),
        padding='same', activation='relu')(x)
    path3 = layers.Conv2D(
        filters_5x5, (1, 1),
        padding='same', activation='relu')(path3)

    path4 = layers.MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    path4 = layers.Conv2D(filters_pool, (1, 1),
                          padding='same', activation='relu')(path4)

    return tf.concat([path1, path2, path3, path4], axis=3)

def prepare_model(height: int = 300, width: int = 300, channels:int = 3, no_augment:bool = True):
    """Perpare model for usage

    :return: Model
    :rtype: tensorflow.keras.Model
    """
    inp = layers.Input(shape=(height, width, channels))
    input_tensor = tf.keras.layers.experimental.preprocessing.Resizing(
        224, 224, interpolation="bilinear")(inp)
    
    # Add the preprocessing/augmentation layers.
    if no_augment:
        x = layers.Conv2D(64, 7, strides=2, padding='same',activation='relu')(input_tensor)
    else:
        x = tf.keras.layers.RandomFlip(mode='horizontal_and_vertical')(input_tensor)
        x = tf.keras.layers.RandomRotation(0.2)(x)
        x = tf.keras.layers.RandomZoom(height_factor=(0.05, 0.15))(x)
        x = tf.keras.layers.RandomContrast(0.2)(x)
        x = tf.keras.layers.Rescaling(1./255)(x)
        # x = tf.keras.layers.Resizing(224,224,interpolation="bilinear")(x)
        x = layers.Conv2D(64, 7, strides=2, padding='same',activation='relu')(x)

    x = layers.MaxPooling2D(3, strides=2)(x)

    x = layers.Conv2D(64, 1, strides=1, padding='same', activation='relu')(x)
    x = layers.Conv2D(192, 3, strides=1, padding='same', activation='relu')(x)

    x = layers.MaxPooling2D(3, strides=2)(x)

    x = inception(
        x,
        filters_1x1=64,
        filters_3x3_reduce=96,
        filters_3x3=128,
        filters_5x5_reduce=16,
        filters_5x5=32,
        filters_pool=32
    )

    x = inception(
        x,
        filters_1x1=128,
        filters_3x3_reduce=128,
        filters_3x3=192,
        filters_5x5_reduce=32,
        filters_5x5=96,
        filters_pool=64
    )

    x = layers.MaxPooling2D(3, strides=2)(x)

    x = inception(
        x,
        filters_1x1=192,
        filters_3x3_reduce=96,
        filters_3x3=208,
        filters_5x5_reduce=16,
        filters_5x5=48,
        filters_pool=64
    )

    aux1 = layers.AveragePooling2D((5, 5), strides=3)(x)
    aux1 = layers.Conv2D(128, 1, padding='same', activation='relu')(aux1)
    aux1 = layers.Flatten()(aux1)
    aux1 = layers.Dense(1024, activation='relu')(aux1)
    aux1 = layers.Dropout(0.7)(aux1)
    aux1 = layers.Dense(4, activation='softmax')(aux1)

    x = inception(
        x,
        filters_1x1=160,
        filters_3x3_reduce=112,
        filters_3x3=224,
        filters_5x5_reduce=24,
        filters_5x5=64,
        filters_pool=64
    )

    x = inception(
        x,
        filters_1x1=128,
        filters_3x3_reduce=128,
        filters_3x3=256,
        filters_5x5_reduce=24,
        filters_5x5=64,
        filters_pool=64
    )

    x = inception(
        x,
        filters_1x1=112,
        filters_3x3_reduce=144,
        filters_3x3=288,
        filters_5x5_reduce=32,
        filters_5x5=64,
        filters_pool=64
    )

    aux2 = layers.AveragePooling2D((5, 5), strides=3)(x)
    aux2 = layers.Conv2D(128, 1, padding='same', activation='relu')(aux2)
    aux2 = layers.Flatten()(aux2)
    aux2 = layers.Dense(1024, activation='relu')(aux2)
    aux2 = layers.Dropout(0.7)(aux2)
    aux2 = layers.Dense(4, activation='softmax')(aux2)

    x = inception(
        x,
        filters_1x1=256,
        filters_3x3_reduce=160,
        filters_3x3=320,
        filters_5x5_reduce=32,
        filters_5x5=128,
        filters_pool=128
    )

    x = layers.MaxPooling2D(3, strides=2)(x)

    x = inception(
        x,
        filters_1x1=256,
        filters_3x3_reduce=160,
        filters_3x3=320,
        filters_5x5_reduce=32,
        filters_5x5=128,
        filters_pool=128
    )

    x = inception(
        x,
        filters_1x1=384,
        filters_3x3_reduce=192,
        filters_3x3=384,
        filters_5x5_reduce=48,
        filters_5x5=128,
        filters_pool=128
    )

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.4)(x)
    out = layers.Dense(4, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out, name="GoogLeNetV1-Aux")


    model.compile(
        optimizer='adam',
        loss=[
            losses.sparse_categorical_crossentropy,
            # losses.sparse_categorical_crossentropy,
            # losses.sparse_categorical_crossentropy
        ],
        # loss_weights=[1, 0.3, 0.3],
        metrics=['accuracy']
    )
    
    return model