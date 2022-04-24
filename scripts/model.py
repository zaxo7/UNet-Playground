import tensorflow as tf

def Conv2D_BN(model, filters, kernel_size=(3, 3), activation='relu', padding='valid', strides=(1, 1)):
    x = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, padding=padding, strides=strides)(model)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)
    return x

def UpConv2D_BN(model, filters, kernel_size=(2, 2), activation='relu', padding='valid', strides=2):
    x = tf.keras.layers.Conv2DTranspose(filters,
                                        kernel_size=kernel_size,
                                        padding=padding,
                                        strides=strides)(model)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)
    return x


def get_do_unet(compile = True):
    np_filters = 32

    inputs = tf.keras.layers.Input((188, 188, 3))
    
    down1 = Conv2D_BN(3*np_filters, kernel_size=(3, 3))(inputs) # 186x186
    down1 = Conv2D_BN(np_filters, kernel_size=(1, 1))(down1)  # 186x186
    down1 = Conv2D_BN(np_filters, kernel_size=(3, 3))(down1)  # 184x184
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(down1)  # 92x92

    np_filters *= 2
    down2 = Conv2D_BN(np_filters, kernel_size=(3, 3))(pool1)  # 90x90
    down2 = Conv2D_BN(np_filters, kernel_size=(3, 3))(down2)  # 88x88
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(down2)  # 44x44

    np_filters *= 2
    down3 = Conv2D_BN(np_filters, kernel_size=(3, 3))(pool2)  # 42x42
    down3 = Conv2D_BN(np_filters, kernel_size=(3, 3))(down3)  # 40x40
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(down3)  # 20x20

    np_filters *= 2
    down4 = Conv2D_BN(np_filters, kernel_size=(3, 3))(pool3)  # 18x18
    down4 = Conv2D_BN(np_filters, kernel_size=(3, 3))(down4)  # 16x16

    np_filters /= 2
    up1 = UpConv2D_BN(np_filters)(down4)  # 32x32
    up1 = tf.keras.layers.concatenate([tf.keras.layers.Cropping2D(4)(down3), up1])  # 32x32
    up1 = Conv2D_BN(np_filters, kernel_size=(3, 3))(up1)  # 30x30
    up1 = Conv2D_BN(np_filters, kernel_size=(3, 3))(up1)  # 28x28

    np_filters /= 2
    up2 = UpConv2D_BN(np_filters)(up1)  #56x56
    up2 = tf.keras.layers.concatenate([tf.keras.layers.Cropping2D(16)(down2), up2])  # 56x56
    up2 = Conv2D_BN(np_filters, kernel_size=(3, 3))(up2)  # 54x54
    up2 = Conv2D_BN(np_filters, kernel_size=(3, 3))(up2)  # 52x52

    np_filters /= 2
    up3 = UpConv2D_BN(np_filters)(up2)  # 104x104
    up3 = tf.keras.layers.concatenate([tf.keras.layers.Cropping2D(40)(down1), up3])  # 104x104
    up3 = Conv2D_BN(np_filters, kernel_size=(3, 3))(up3)  # 102x102
    up3 = Conv2D_BN(np_filters, kernel_size=(3, 3))(up3)  # 100x100

    out_mask = tf.keras.layers.Conv2D(1, (1, 1), activation='softmax', name='mask')(up3)
    out_edge = tf.keras.layers.Conv2D(1, (1, 1), activation='softmax', name='edge')(up3)

    model = tf.keras.models.Model(inputs=inputs, outputs=(out_mask, out_edge))

    return model