import tensorflow as tf


def create_model(input_shape):
    # See https://towardsdatascience.com/implementing-alexnet-cnn-architecture-using-tensorflow-2-0-and-keras-2113e090ad98
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv3D(
            filters=96, kernel_size=(11, 11, 11), strides=(4, 4, 4), activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2)),
        tf.keras.layers.Conv3D(
            filters=256, kernel_size=(5, 5, 5), strides=(1, 1, 1), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2)),
        tf.keras.layers.Conv3D(
            filters=384, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv3D(
            filters=384, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv3D(
            filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu')
    ])

    return model
