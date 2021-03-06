"""Run the model"""
import glob

import model


def run_training(model_name):
    train_img_files = glob.glob('data/train/*.jpg')
    test_img_files = glob.glob('data/test/*.jpg')

    do_unet = model.DO_UNet(train_img_files,
                            test_img_files,
                            scale_invariant=True)

    do_unet.fit(model_name,
                epochs=1,
                imgs_per_epoch=64,
                batchsize=8,
                workers=8)
    
    return do_unet


if __name__ == '__main__':
    run_training('Test_scale')
