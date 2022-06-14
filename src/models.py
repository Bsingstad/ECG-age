from tensorflow import keras

def attia_network_age(samp_freq,time,num_leads):
    input_layer = keras.layers.Input(shape=(samp_freq*time, num_leads))


    # Temporal analysis block 1
    conv1 = keras.layers.Conv1D(filters=16,kernel_size=7,strides=1,padding='same')(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation("relu")(conv1)
    conv1 = keras.layers.MaxPooling1D(pool_size=2)(conv1)
    
    # Temporal analysis block 2
    conv2 = keras.layers.Conv1D(filters=16,kernel_size=5,strides=1,padding='same')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation("relu")(conv2)
    conv2 = keras.layers.MaxPooling1D(pool_size=2)(conv2)
    
    # Temporal analysis block 3
    conv3 = keras.layers.Conv1D(filters=32,kernel_size=5,strides=1,padding='same')(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation("relu")(conv3)
    conv3 = keras.layers.MaxPooling1D(pool_size=2)(conv3)
    
    # Temporal analysis block 4
    conv4 = keras.layers.Conv1D(filters=32,kernel_size=5,strides=1,padding='same')(conv3)
    conv4 = keras.layers.BatchNormalization()(conv4)
    conv4 = keras.layers.Activation("relu")(conv4)
    conv4 = keras.layers.MaxPooling1D(pool_size=2)(conv4)
    
    # Temporal analysis block 5
    conv5 = keras.layers.Conv1D(filters=64,kernel_size=5,strides=1,padding='same')(conv4)
    conv5 = keras.layers.BatchNormalization()(conv5)
    conv5 = keras.layers.Activation("relu")(conv5)
    conv5 = keras.layers.MaxPooling1D(pool_size=2)(conv5)
    
    # Temporal analysis block 6
    conv6 = keras.layers.Conv1D(filters=64,kernel_size=3,strides=1,padding='same')(conv5)
    conv6 = keras.layers.BatchNormalization()(conv6)
    conv6 = keras.layers.Activation("relu")(conv6)
    conv6 = keras.layers.MaxPooling1D(pool_size=2)(conv6)
    
    # Temporal analysis block 7
    conv7 = keras.layers.Conv1D(filters=64,kernel_size=3,strides=1,padding='same')(conv6)
    conv7 = keras.layers.BatchNormalization()(conv7)
    conv7 = keras.layers.Activation("relu")(conv7)
    conv7 = keras.layers.MaxPooling1D(pool_size=2)(conv7)
    
    # Temporal analysis block 8
    conv8 = keras.layers.Conv1D(filters=64,kernel_size=3,strides=1,padding='same')(conv7)
    conv8 = keras.layers.BatchNormalization()(conv8)
    conv8 = keras.layers.Activation("relu")(conv8)
    conv8 = keras.layers.MaxPooling1D(pool_size=2)(conv8)
    
    # Spatial analysis block 1
    spatial_block_1 = keras.layers.Conv1D(filters=128,kernel_size=1,strides=1,padding='same')(conv8)
    spatial_block_1 = keras.layers.BatchNormalization()(spatial_block_1)
    spatial_block_1 = keras.layers.Activation("relu")(spatial_block_1)
    spatial_block_1 = keras.layers.MaxPooling1D(pool_size=2)(spatial_block_1)
    spatial_block_1 = keras.layers.Flatten()(spatial_block_1)
    
    # Fully Connected block 1
    fc_block_1 = keras.layers.Dense(units=128)(spatial_block_1)
    fc_block_1 = keras.layers.BatchNormalization()(fc_block_1)
    fc_block_1 = keras.layers.Activation("relu")(fc_block_1)
    fc_block_1 = keras.layers.Dropout(rate=0.2)(fc_block_1)
    
    # Fully Connected block 1
    fc_block_2 = keras.layers.Dense(units=64)(fc_block_1)
    fc_block_2 = keras.layers.BatchNormalization()(fc_block_2)
    fc_block_2 = keras.layers.Activation("relu")(fc_block_2)
    fc_block_2 = keras.layers.Dropout(rate=0.2)(fc_block_2)   
                  
    

    #output_layer_1 = keras.layers.Dense(units=1,activation='linear')(last_dense)
    output_layer = keras.layers.Dense(units=1,activation='linear')(fc_block_2)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=[tf.keras.metrics.MeanAbsoluteError()])

    return model


def _inception_module(input_tensor, stride=1, activation='linear', use_bottleneck=True, kernel_size=40, bottleneck_size=32, nb_filters=32):

    if use_bottleneck and int(input_tensor.shape[-1]) > 1:
        input_inception = keras.layers.Conv1D(filters=bottleneck_size, kernel_size=1,
                                              padding='same', activation=activation, use_bias=False)(input_tensor)
    else:
        input_inception = input_tensor

    # kernel_size_s = [3, 5, 8, 11, 17]
    kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]

    conv_list = []

    for i in range(len(kernel_size_s)):
        conv_list.append(keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size_s[i],
                                              strides=stride, padding='same', activation=activation, use_bias=False)(
            input_inception))

    max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

    conv_6 = keras.layers.Conv1D(filters=nb_filters, kernel_size=1,
                                  padding='same', activation=activation, use_bias=False)(max_pool_1)

    conv_list.append(conv_6)

    x = keras.layers.Concatenate(axis=2)(conv_list)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    return x

def _shortcut_layer(input_tensor, out_tensor):
    shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                      padding='same', use_bias=False)(input_tensor)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    x = keras.layers.Add()([shortcut_y, out_tensor])
    x = keras.layers.Activation('relu')(x)
    return x

def build_model(input_shape, nb_classes, depth=25, use_residual=True):
    input_layer = keras.layers.Input(input_shape)

    x = input_layer
    input_res = input_layer

    for d in range(depth):

        x = _inception_module(x)

        if use_residual and d % 3 == 2:
            x = _shortcut_layer(input_res, x)
            input_res = x

    gap_layer = keras.layers.GlobalAveragePooling1D()(x)

    output_layer = keras.layers.Dense(units=nb_classes,activation='linear')(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    #model.compile(loss=[macro_double_soft_f1], optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    model.compile(loss=keras.losses.MeanAbsoluteError(), optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=[keras.metrics.MeanSquaredError()])
    
    return model

