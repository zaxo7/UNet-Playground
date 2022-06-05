from numpy import full
import tensorflow as tf
import numpy as np

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


def get_do_unet(compile = True):
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

    out_mask = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='mask')(up3)
    out_edge = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='edge')(up3)
    

    model = tf.keras.models.Model(inputs=inputs, outputs=(out_mask, out_edge))
    
    if compile:
        model.compile(optimizer="adam",
             loss="binary_crossentropy",
             loss_weights=[0.3, 0.7],
             metrics={'mask': [mean_iou, dsc, tversky, 'acc'], 
                      'edge': [mean_iou, dsc, tversky, 'acc']})

    return model

def get_do_unet_debug():
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

    out_mask = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='mask')(up3)
    out_edge = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='edge')(up3)

    model = tf.keras.models.Model(inputs=inputs, outputs=(out_mask, out_edge, pool1, pool2, pool3, down4, up1, up2, up3))

    return model



def generate_train_dataset(img_files):
        imgs, mask, edge = data.load_data(img_files)
        
        print(f"mask :  {np.unique(mask[0], return_counts=True)}")
        print(f"edge :  {np.unique(edge[0], return_counts=True)}")
        
        print(f"mask :  {mask[0].shape}")
        print(f"edge :  {edge[0].shape}")
        print(f"img :  {imgs[0].shape}")

        def train_gen():
            return data.train_generator(imgs, mask,
                                        edge=edge,
                                        padding=100,
                                        input_size=188,
                                        output_size=100)


        return train_gen()
    
def generate_train_dataset3(img_files, mask_files, edge_files):
        imgs = data.load_data_na(img_files, RGB=True, clahe=True)
        mask = data.load_data_na(mask_files)
        edge = data.load_data_na(edge_files)
        

        print(f"mask :  {np.unique(mask[0], return_counts=True)}")
        print(f"edge :  {np.unique(edge[0], return_counts=True)}")
        #print(f"img :  {np.unique(imgs[0], return_counts=True)}")
        
        print(f"mask :  {mask[0].shape}")
        print(f"edge :  {edge[0].shape}")
        print(f"img :  {imgs[0].shape}")


        def train_gen():
            return data.train_generator(imgs, mask,
                                        edge=edge,
                                        padding=100,
                                        input_size=188,
                                        output_size=100)


        return train_gen()
    
    
def generate_train_dataset3_tf(img_files, mask_files, edge_files):
        imgs = data.load_data_na(img_files, RGB=True, clahe=True)
        mask = data.load_data_na(mask_files)
        edge = data.load_data_na(edge_files)
        

        print(f"mask :  {np.unique(mask[0], return_counts=True)}")
        print(f"edge :  {np.unique(edge[0], return_counts=True)}")
        #print(f"img :  {np.unique(imgs[0], return_counts=True)}")
        
        print(f"mask :  {mask[0].shape}")
        print(f"edge :  {edge[0].shape}")
        print(f"img :  {imgs[0].shape}")


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
        
        
        
def generate_train_dataset_tf(img_files):
        imgs, mask, edge = data.load_data(img_files)
        
        print(f"mask :  {np.unique(mask[0], return_counts=True)}")
        print(f"edge :  {np.unique(edge[0], return_counts=True)}")
        
        print(f"mask :  {mask[0].shape}")
        print(f"edge :  {edge[0].shape}")
        print(f"img :  {imgs[0].shape}")

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
    
    
def generate_test_dataset_tf(img_files):
    imgs, mask, edge = data.load_data(img_files)

    img_chips, mask_chips, edge_chips = data.test_chips(imgs, mask,
                                                        edge=edge,
                                                        padding=100,
                                                        input_size=188,
                                                        output_size=100)
    

    return tf.data.Dataset.from_tensor_slices((img_chips,
                                                (mask_chips, edge_chips))
                                                )
    

def generate_test_dataset3(img_files, mask_files, edge_files):
    imgs = data.load_data_na(img_files, RGB=True, clahe=True)
    mask = data.load_data_na(mask_files)
    edge = data.load_data_na(edge_files)

    img_chips, mask_chips, edge_chips = data.test_chips(imgs, mask,
                                                        edge=edge,
                                                        padding=100,
                                                        input_size=188,
                                                        output_size=100)
    
    return img_chips, mask_chips, edge_chips

    
def generate_test_dataset3_tf(img_files, mask_files, edge_files):
    imgs = data.load_data_na(img_files, RGB=True, clahe=True)
    mask = data.load_data_na(mask_files)
    edge = data.load_data_na(edge_files)

    img_chips, mask_chips, edge_chips = data.test_chips(imgs, mask,
                                                        edge=edge,
                                                        padding=100,
                                                        input_size=188,
                                                        output_size=100)
    

    return tf.data.Dataset.from_tensor_slices((img_chips,
                                                (mask_chips, edge_chips))
                                                )
    
    
def predictFullImage(model,
                    imgs,
                    padding=100,
                    input_size=188,
                    output_size=100,
                    normalize_input = False,
                    normalize_output = False):
    
    if normalize_input:
        imgs = data.clahe_images(imgs)
    
    images = imgs
    masks = []
    edges = []
    
    for img in imgs:
        slices, img_sizes = data.slice_images([img],
                padding=padding,
                input_size=input_size,
                output_size=output_size)
        
          
        prediction = model.predict(slices)
        
        mask_slices = prediction[0]
        edge_slices = prediction[1]
        
        mask_slices_2d = mask_slices.reshape((img_sizes[0][0], img_sizes[0][1], 100, 100, 1))
        edge_slices_2d = edge_slices.reshape((img_sizes[0][0], img_sizes[0][1], 100, 100, 1))
        
        full_mask = data.concat_slices(mask_slices_2d)
        full_edge = data.concat_slices(edge_slices_2d)
        
        if normalize_output:
            full_mask = (full_mask > 0.5) * 1
            full_edge = (full_edge > 0.5) * 1
        
        masks += [full_mask]
        edges += [full_edge]
        
    return (images, masks, edges)
    
    
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
    
@tf.function
def dsc(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) +
                                            tf.reduce_sum(y_pred_f) +
                                            smooth)


@tf.function
def dice_loss(y_true, y_pred):
    return 1 - dsc(y_true, y_pred)


@tf.function
def tversky(y_true, y_pred):
    alpha = 0.7
    smooth = 1.0
    y_true_pos = tf.reshape(y_true, [-1])
    y_pred_pos = tf.reshape(y_pred, [-1])
    true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
    false_neg = tf.reduce_sum(y_true_pos * (1 - y_pred_pos))
    false_pos = tf.reduce_sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

@tf.function
def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)

# in this funciton i am adding the binarisation of results
@tf.function
def tversky_b(y_true, y_pred):
    y_pred = (y_pred > 0.5)
    y_pred = tf.cast(y_pred, tf.float32)
    alpha = 0.7
    smooth = 1.0
    y_true_pos = tf.reshape(y_true, [-1])
    y_pred_pos = tf.reshape(y_pred, [-1])
    true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
    false_neg = tf.reduce_sum(y_true_pos * (1 - y_pred_pos))
    false_pos = tf.reduce_sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

@tf.function
def tversky_loss_b(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)




@tf.function
def focal_tversky(y_true, y_pred):
    return tf.pow((1 - tversky(y_true, y_pred)), 0.75)


@tf.function
def iou(y_true, y_pred):
    intersect = tf.reduce_sum(y_true * y_pred, axis=(1, 2))
    union = tf.reduce_sum(y_true + y_pred, axis=(1, 2))
    return tf.reduce_mean(tf.math.divide_no_nan(intersect, (union - intersect)), axis=1)


@tf.function
def mean_iou(y_true, y_pred):
    y_true_32 = tf.cast(y_true, tf.float32)
    y_pred_32 = tf.cast(y_pred, tf.float32)
    score = tf.map_fn(lambda x: iou(y_true_32, tf.cast(y_pred_32 > x, tf.float32)),
                      tf.range(0.5, 1.0, 0.05, tf.float32),
                      tf.float32)
    return tf.reduce_mean(score)


@tf.function
def iou_loss(y_true, y_pred):
    return -1*mean_iou(y_true, y_pred)




@tf.function
def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
    targets = tf.cast(targets, tf.float32)
    weight_a = alpha * (1 - y_pred) ** gamma * targets
    weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)

    return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b 

@tf.function
def focal_loss(y_true, logits):
    alpha=0.25
    gamma=2
    y_pred = tf.math.sigmoid(logits)
    loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)

    return tf.reduce_mean(loss)

