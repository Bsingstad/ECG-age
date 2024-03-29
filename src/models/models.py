import tensorflow as tf
from typing import Tuple


def attia_network_age(samp_freq: int, time: int, num_leads: int) -> tf.keras.models.Model:
    """The model proposed by Attia et al. 2019"""
    input_layer = tf.keras.layers.Input(shape=(samp_freq * time, num_leads))

    # Temporal analysis block 1
    conv1 = tf.keras.layers.Conv1D(
        filters=16, kernel_size=7, strides=1, padding="same"
    )(input_layer)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.Activation("relu")(conv1)
    conv1 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv1)

    # Temporal analysis block 2
    conv2 = tf.keras.layers.Conv1D(
        filters=16, kernel_size=5, strides=1, padding="same"
    )(conv1)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = tf.keras.layers.Activation("relu")(conv2)
    conv2 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv2)

    # Temporal analysis block 3
    conv3 = tf.keras.layers.Conv1D(
        filters=32, kernel_size=5, strides=1, padding="same"
    )(conv2)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    conv3 = tf.keras.layers.Activation("relu")(conv3)
    conv3 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv3)

    # Temporal analysis block 4
    conv4 = tf.keras.layers.Conv1D(
        filters=32, kernel_size=5, strides=1, padding="same"
    )(conv3)
    conv4 = tf.keras.layers.BatchNormalization()(conv4)
    conv4 = tf.keras.layers.Activation("relu")(conv4)
    conv4 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv4)

    # Temporal analysis block 5
    conv5 = tf.keras.layers.Conv1D(
        filters=64, kernel_size=5, strides=1, padding="same"
    )(conv4)
    conv5 = tf.keras.layers.BatchNormalization()(conv5)
    conv5 = tf.keras.layers.Activation("relu")(conv5)
    conv5 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv5)

    # Temporal analysis block 6
    conv6 = tf.keras.layers.Conv1D(
        filters=64, kernel_size=3, strides=1, padding="same"
    )(conv5)
    conv6 = tf.keras.layers.BatchNormalization()(conv6)
    conv6 = tf.keras.layers.Activation("relu")(conv6)
    conv6 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv6)

    # Temporal analysis block 7
    conv7 = tf.keras.layers.Conv1D(
        filters=64, kernel_size=3, strides=1, padding="same"
    )(conv6)
    conv7 = tf.keras.layers.BatchNormalization()(conv7)
    conv7 = tf.keras.layers.Activation("relu")(conv7)
    conv7 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv7)

    # Temporal analysis block 8
    conv8 = tf.keras.layers.Conv1D(
        filters=64, kernel_size=3, strides=1, padding="same"
    )(conv7)
    conv8 = tf.keras.layers.BatchNormalization()(conv8)
    conv8 = tf.keras.layers.Activation("relu")(conv8)
    conv8 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv8)

    # Spatial analysis block 1
    spatial_block_1 = tf.keras.layers.Conv1D(
        filters=128, kernel_size=1, strides=1, padding="same"
    )(conv8)
    spatial_block_1 = tf.keras.layers.BatchNormalization()(spatial_block_1)
    spatial_block_1 = tf.keras.layers.Activation("relu")(spatial_block_1)
    spatial_block_1 = tf.keras.layers.MaxPooling1D(pool_size=2)(spatial_block_1)
    spatial_block_1 = tf.keras.layers.Flatten()(spatial_block_1)

    # Fully Connected block 1
    fc_block_1 = tf.keras.layers.Dense(units=128)(spatial_block_1)
    fc_block_1 = tf.keras.layers.BatchNormalization()(fc_block_1)
    fc_block_1 = tf.keras.layers.Activation("relu")(fc_block_1)
    fc_block_1 = tf.keras.layers.Dropout(rate=0.2)(fc_block_1)

    # Fully Connected block 1
    fc_block_2 = tf.keras.layers.Dense(units=64)(fc_block_1)
    fc_block_2 = tf.keras.layers.BatchNormalization()(fc_block_2)
    fc_block_2 = tf.keras.layers.Activation("relu")(fc_block_2)
    fc_block_2 = tf.keras.layers.Dropout(rate=0.2)(fc_block_2)

    # output_layer_1 = tf.keras.layers.Dense(units=1,activation='linear')(last_dense)
    output_layer = tf.keras.layers.Dense(units=1, activation="linear")(fc_block_2)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=[tf.keras.metrics.MeanAbsoluteError()],
    )

    return model


def _inception_module(
    input_tensor,
    stride=1,
    activation="linear",
    use_bottleneck=True,
    kernel_size=40,
    bottleneck_size=32,
    nb_filters=32,
):

    if use_bottleneck and int(input_tensor.shape[-1]) > 1:
        input_inception = tf.keras.layers.Conv1D(
            filters=bottleneck_size,
            kernel_size=1,
            padding="same",
            activation=activation,
            use_bias=False,
        )(input_tensor)
    else:
        input_inception = input_tensor

    # kernel_size_s = [3, 5, 8, 11, 17]
    kernel_size_s = [kernel_size // (2**i) for i in range(3)]

    conv_list = []

    for i in range(len(kernel_size_s)):
        conv_list.append(
            tf.keras.layers.Conv1D(
                filters=nb_filters,
                kernel_size=kernel_size_s[i],
                strides=stride,
                padding="same",
                activation=activation,
                use_bias=False,
            )(input_inception)
        )

    max_pool_1 = tf.keras.layers.MaxPool1D(pool_size=3, strides=stride, padding="same")(
        input_tensor
    )

    conv_6 = tf.keras.layers.Conv1D(
        filters=nb_filters,
        kernel_size=1,
        padding="same",
        activation=activation,
        use_bias=False,
    )(max_pool_1)

    conv_list.append(conv_6)

    x = tf.keras.layers.Concatenate(axis=2)(conv_list)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation="relu")(x)
    return x


def _shortcut_layer(input_tensor, out_tensor):
    shortcut_y = tf.keras.layers.Conv1D(
        filters=int(out_tensor.shape[-1]), kernel_size=1, padding="same", use_bias=False
    )(input_tensor)
    shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)

    x = tf.keras.layers.Add()([shortcut_y, out_tensor])
    x = tf.keras.layers.Activation("relu")(x)
    return x


def build_model(
    input_shape: Tuple[int, int],
    nb_classes: int,
    depth: int = 6,
    use_residual: bool = True,
)-> tf.keras.models.Model:
    """
    Model proposed by HI Fawas et al 2019 "Finding AlexNet for Time Series Classification - InceptionTime"
    """
    input_layer = tf.keras.layers.Input(input_shape)

    x = input_layer
    input_res = input_layer

    for d in range(depth):

        x = _inception_module(x)

        if use_residual and d % 3 == 2:
            x = _shortcut_layer(input_res, x)
            input_res = x

    gap_layer = tf.keras.layers.GlobalAveragePooling1D()(x)

    output_layer = tf.keras.layers.Dense(units=nb_classes, activation="linear")(
        gap_layer
    )

    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        loss=tf.keras.losses.MeanAbsoluteError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=[tf.keras.metrics.MeanSquaredError()],
    )

    return model
