import keras as keras
import keras.layers

input_shape = [176, 256, 256, 1]


def build_alex_net():
    model = keras.models.Sequential([
        # In and C1
        keras.layers.Conv3D(filters=96, kernel_size=11, activation="relu", padding="valid",
                            input_shape=input_shape, strides=4),
        # S2
        keras.layers.MaxPooling3D(pool_size=3, strides=2, padding="valid"),
        # C3
        keras.layers.Conv3D(filters=256, kernel_size=5, strides=1, padding="same", activation="relu"),
        # S4
        keras.layers.MaxPooling3D(pool_size=3, strides=2, padding="valid"),
        # C5
        keras.layers.Conv3D(filters=256, kernel_size=3, strides=1, padding="same", activation="relu"),
        # C6
        keras.layers.Conv3D(filters=256, kernel_size=3, strides=1, padding="same", activation="relu"),
        # C7
        keras.layers.Conv3D(filters=256, kernel_size=3, strides=1, padding="same", activation="relu"),
        # F8
        keras.layers.Dense(units=4096),
        # F9
        keras.layers.Dense(units=4096),
        # Out
        keras.layers.Dense(units=1000, activation='linear'),
    ])

    return model


alex_net = build_alex_net()
print(alex_net.summary())
