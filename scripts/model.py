import tensorflow as tf

import data

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


def get_do_unet():
    np_filters = 32

    inputs = tf.keras.layers.Input((188, 188, 3))
    
    down1 = Conv2D_BN(inputs, filters=3*np_filters, kernel_size=(3, 3)) # 186x186
    down1 = Conv2D_BN(down1, filters=np_filters, kernel_size=(1, 1))  # 186x186
    down1 = Conv2D_BN(down1, filters=np_filters, kernel_size=(3, 3))  # 184x184
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(down1)  # 92x92

    np_filters *= 2
    down2 = Conv2D_BN(pool1, filters=np_filters, kernel_size=(3, 3))  # 90x90
    down2 = Conv2D_BN(down2, filters=np_filters, kernel_size=(3, 3))  # 88x88
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(down2)  # 44x44

    np_filters *= 2
    down3 = Conv2D_BN(pool2, filters=np_filters, kernel_size=(3, 3))  # 42x42
    down3 = Conv2D_BN(down3, filters=np_filters, kernel_size=(3, 3))  # 40x40
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(down3)  # 20x20

    np_filters *= 2
    down4 = Conv2D_BN(pool3, filters=np_filters, kernel_size=(3, 3))  # 18x18
    down4 = Conv2D_BN(down4, filters=np_filters, kernel_size=(3, 3))  # 16x16

    np_filters /= 2
    up1 = UpConv2D_BN(down4, filters=np_filters)  # 32x32
    up1 = tf.keras.layers.concatenate([tf.keras.layers.Cropping2D(4)(down3), up1])  # 32x32
    up1 = Conv2D_BN(up1, filters=np_filters, kernel_size=(3, 3))  # 30x30
    up1 = Conv2D_BN(up1, filters=np_filters, kernel_size=(3, 3))  # 28x28

    np_filters /= 2
    up2 = UpConv2D_BN(up1, filters=np_filters)  #56x56
    up2 = tf.keras.layers.concatenate([tf.keras.layers.Cropping2D(16)(down2), up2])  # 56x56
    up2 = Conv2D_BN(up2, filters=np_filters, kernel_size=(3, 3))  # 54x54
    up2 = Conv2D_BN(up2, filters=np_filters, kernel_size=(3, 3))  # 52x52

    np_filters /= 2
    up3 = UpConv2D_BN(up2, filters=np_filters)  # 104x104
    up3 = tf.keras.layers.concatenate([tf.keras.layers.Cropping2D(40)(down1), up3])  # 104x104
    up3 = Conv2D_BN(up3, np_filters, kernel_size=(3, 3))  # 102x102
    up3 = Conv2D_BN(up3, np_filters, kernel_size=(3, 3))  # 100x100

    out_mask = tf.keras.layers.Conv2D(1, (1, 1), activation='softmax', name='mask')(up3)
    out_edge = tf.keras.layers.Conv2D(1, (1, 1), activation='softmax', name='edge')(up3)

    model = tf.keras.models.Model(inputs=inputs, outputs=(out_mask, out_edge))

    return model



def generate_train_dataset(img_files):
        imgs, mask, edge = data.load_data(img_files)

        def train_gen():
            return data.train_generator(imgs, mask,
                                        edge=edge,
                                        padding=100,
                                        input_size=188,
                                        output_size=100)


        return train_gen()
        
def generate_train_dataset_tf(img_files):
        imgs, mask, edge = data.load_data(img_files)

        def train_gen():
            return data.train_generator(imgs, mask,
                                        edge=edge,
                                        padding=100,
                                        input_size=188,
                                        output_size=100)

        return tf.data.Dataset.from_generator(train_gen,
                                              (tf.float64, ((tf.float64), (tf.float64))),
                                              ((188, 188, 3), ((100, 100, 1), (100, 100, 1)))
                                             )


def generate_test_dataset(img_files):
    imgs, mask, edge = data.load_data(img_files)

    img_chips, mask_chips, edge_chips = data.test_chips(imgs, mask,
                                                        edge=edge,
                                                        padding=100,
                                                        input_size=188,
                                                        output_size=100)
    
    return img_chips, mask_chips, edge_chips

    return tf.data.Dataset.from_tensor_slices((img_chips,
                                                (mask_chips, edge_chips))
                                                )
    
    
    
def get_callbacks(name):
    return [
        tf.keras.callbacks.ModelCheckpoint(f'models/{name}_all.h5',
                                           save_best_only=False,
                                           save_weights_only=True,
                                           verbose=0),
        tf.keras.callbacks.ModelCheckpoint(f'models/{name}_best.h5',
                                           save_best_only=True,
                                           save_weights_only=True,
                                           verbose=1)
    ]
    
