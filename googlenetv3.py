import tensorflow as tf


def prepare_model(height: int = 300, width: int = 300, channels: int = 3):
    inp = tf.keras.layers.Input(shape=(height, width, channels))
    # greyscale = tf.image.rgb_to_grayscale(inp)
    input_tensor = tf.keras.layers.experimental.preprocessing.Resizing(
        224, 224, 
        interpolation="bilinear"
    )(inp)

    
    model = tf.keras.applications.inception_v3.InceptionV3(
        include_top=True,
        weights=None,
        classes=4,
        input_tensor=input_tensor,
    )
  
    model.compile(
        optimizer="rmsprop",
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy']
    )
  
    return model