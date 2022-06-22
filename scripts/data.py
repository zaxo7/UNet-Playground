"""Load and preprocess data"""
from operator import contains
import os
import json


import cv2
import numpy as np
from tensorflow import keras
from PIL import Image
from tqdm import tqdm
import json


import matplotlib.pyplot as plt


def load_image_list(img_files, RGB=False):
    imgs = []
    for image_file in img_files:
        if RGB:
            img = cv2.imread(image_file)
        else:
            img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            thresh = 127
            img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
            
        imgs += [img]
    return imgs


def clahe_images(img_list):
    for i, img in enumerate(img_list):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[..., 0] = clahe.apply(lab[..., 0])
        img_list[i] = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return img_list


def get_label(word):
    words_to_numerical_labels_dict = {
        'None': 0,
        'Truck': 1,
        'Tanker': 2,
        'Trailer': 3
    }
    return words_to_numerical_labels_dict[word]


def make_polygon_lists(markup_files):
    marked_dicts = []
    for file_name in markup_files:
        if os.path.isfile(file_name):
            with open(file_name, 'r') as markup_file:
                marked_dicts += [json.load(markup_file)]

    polygon_lists_list = []
    for marked_dict in marked_dicts:
        polygon_list = []

        for poly in marked_dict['markup']:
            vertex_list = []

            for vertex in poly['vertices']:
                vertex_list += [(vertex['x'], vertex['y'])]
            polygon_list += [(vertex_list, get_label(poly['object_label']))]

        polygon_lists_list += [polygon_list]

    return polygon_lists_list


def rasterise_markup(polygon_lists_list, imgs, edge_size=None):
    markups = [np.zeros(img.shape[:2]) for img in imgs]
    if edge_size is not None:
        edges = [np.zeros(img.shape[:2]) for img in imgs]

    for i, polygon_list in enumerate(polygon_lists_list):
        for ver, col in polygon_list:
            ver = np.array([ver,])
            cv2.fillPoly(markups[i], ver, col)
            if edge_size is not None:
                cv2.polylines(edges[i], ver, True, col, edge_size)

    if edge_size is not None:
        return markups, edges

    return markups


def load_markup(markup_files, imgs, edge_size=2):
    polygon_list = make_polygon_lists(markup_files)

    mask, edge = rasterise_markup(polygon_list, imgs, edge_size=edge_size)
    mask = [markup.astype(np.uint8) for markup in mask]
    edge = [markup.astype(np.uint8) for markup in edge]

    return mask, edge


# adds padding from (top, bottom), (right, left)
def preprocess_data(imgs, mask, edge, padding=200):
    imgs = [np.pad(img, ((padding, padding),
                         (padding, padding), (0, 0)), mode='constant') for img in imgs]
    mask = [np.pad(mask, ((padding, padding),
                          (padding, padding)), mode='constant') for mask in mask]
    edge = [np.pad(edge, ((padding, padding),
                          (padding, padding)), mode='constant') for edge in edge]

    return imgs, mask, edge

def preprocess_data_na(imgs, padding=200, RGB=False):
    if RGB:
        imgs = [np.pad(img, ((padding, padding),
                         (padding, padding), (0, 0)), mode='constant') for img in imgs]
    else:
        imgs = [np.pad(img, ((padding, padding),
                         (padding, padding)), mode='constant') for img in imgs]

    return imgs


def load_data(img_list, edge_size=2, padding=200):
    imgs = load_image_list(img_list, RGB=True)
    imgs = clahe_images(imgs)

    markup_list = [f.split('.')[0] + '.json' for f in img_list]
    mask, edge = load_markup(markup_list, imgs, edge_size=edge_size)
    
    #print("load_data before preprocessing:")
    #print(f"image: {imgs[0].shape}\nmask : {mask[0].shape}\nedge : {edge[0].shape}")
    
    imgs, mask, edge = preprocess_data(imgs, mask, edge, padding=padding)
    
    #print("load_data after preprocessing:")
    #print(f"image: {imgs[0].shape}\nmask : {mask[0].shape}\nedge : {edge[0].shape}")

    return imgs, mask, edge

def load_data3(img_files, mask_files, edge_files=None, edge_size=2, padding=200, preprocess=True):
    imgs = load_data_na(img_files, RGB=True, clahe=True, preprocess=preprocess)
    mask = load_data_na(mask_files, preprocess=preprocess)
    if edge_files is not None:
        edge = load_data_na(edge_files, preprocess=preprocess)
    
    if edge_files is not None:
        return imgs, mask, edge
    
    return imgs, mask
    

def load_data_na(img_list, edge_size=2, padding=200, RGB=False, clahe=False, preprocess=True):
    img_list = load_image_list(img_list, RGB=RGB)
    if clahe:
        img_list = clahe_images(img_list)
    
    if preprocess:
        img_list = preprocess_data_na(img_list, padding=padding, RGB=RGB)

    return img_list


def aug_lum(image, factor=None):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float64)

    if factor is None:
        lum_offset = 0.5 + np.random.uniform()
    else:
        lum_offset = factor

    hsv[..., 2] = hsv[..., 2] * lum_offset
    hsv[..., 2][hsv[..., 2] > 255] = 255
    hsv = hsv.astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def aug_img(image, calln = 0):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float64)

    hue_offset = 0.8 + 0.4*np.random.uniform()
    sat_offset = 0.5 + np.random.uniform()
    lum_offset = 0.5 + np.random.uniform()

    hsv[..., 0] = hsv[..., 0] * hue_offset
    hsv[..., 1] = hsv[..., 1] * sat_offset
    hsv[..., 2] = hsv[..., 2] * lum_offset

    hsv[..., 0][hsv[..., 0] > 255] = 255
    hsv[..., 1][hsv[..., 1] > 255] = 255
    hsv[..., 2][hsv[..., 2] > 255] = 255

    hsv = hsv.astype(np.uint8)
    
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    #check if the image is fully black don't return it
    pimg = Image.fromarray(img , 'RGB')
    # image all black or all white
    if (sum(pimg.convert("L").getextrema()) in (0, 2)) and (calln < 4):
        return aug_img(image, calln = calln + 1)
    else:
        return img


def train_generator(imgs, mask, edge,
                    scale_range=None,
                    padding=200,
                    input_size=380,
                    output_size=196,
                    skip_empty=False):
    if scale_range is not None:
        scale_range = [1 - scale_range, 1 + scale_range]
    while True:
        # Select which type of cell to return
        chip_type = np.random.choice([True, False])

        while True:
            # Pick random image
            i = np.random.randint(len(imgs))

            # Pick random central location in the image (200 + 196/2)
            center_offset = padding + (output_size / 2)
            x = np.random.randint(center_offset, imgs[i].shape[0] - center_offset)
            y = np.random.randint(center_offset, imgs[i].shape[1] - center_offset)

            # scale the box randomly from x0.8 - 1.2x original size
            scale = 1
            if scale_range is not None:
                scale = scale_range[0] + ((scale_range[0] - scale_range[0]) * np.random.random())

            # find the edges of a box around the image chip and the mask chip
            chip_x_l = int(x - ((input_size / 2) * scale))
            chip_x_r = int(x + ((input_size / 2) * scale))
            chip_y_l = int(y - ((input_size / 2) * scale))
            chip_y_r = int(y + ((input_size / 2) * scale))

            mask_x_l = int(x - ((output_size / 2) * scale))
            mask_x_r = int(x + ((output_size / 2) * scale))
            mask_y_l = int(y - ((output_size / 2) * scale))
            mask_y_r = int(y + ((output_size / 2) * scale))

            # take a slice of the image and mask accordingly
            temp_chip = imgs[i][chip_x_l:chip_x_r, chip_y_l:chip_y_r]
            temp_mask = mask[i][mask_x_l:mask_x_r, mask_y_l:mask_y_r]
            temp_edge = edge[i][mask_x_l:mask_x_r, mask_y_l:mask_y_r]

            if skip_empty:
                if ((temp_mask > 0).sum() > 5) is chip_type:
                    continue

            # resize the image chip back to 380 and the mask chip to 196
            temp_chip = cv2.resize(temp_chip,
                                   (input_size, input_size),
                                   interpolation=cv2.INTER_CUBIC)
            temp_mask = cv2.resize(temp_mask,
                                   (output_size, output_size),
                                   interpolation=cv2.INTER_NEAREST)
            temp_edge = cv2.resize(temp_edge,
                                   (output_size, output_size),
                                   interpolation=cv2.INTER_NEAREST)

            # randomly rotate (like below)
            rot = np.random.randint(4)
            temp_chip = np.rot90(temp_chip, k=rot, axes=(0, 1))
            temp_mask = np.rot90(temp_mask, k=rot, axes=(0, 1))
            temp_edge = np.rot90(temp_edge, k=rot, axes=(0, 1))

            # randomly flip
            if np.random.random() > 0.5:
                temp_chip = np.flip(temp_chip, axis=1)
                temp_mask = np.flip(temp_mask, axis=1)
                temp_edge = np.flip(temp_edge, axis=1)

            # randomly luminosity augment
            temp_chip = aug_img(temp_chip)

            # rescale the image // normalisation to [-1,1] range
            temp_chip = temp_chip.astype(np.float32) * 2
            temp_chip /= 255
            temp_chip -= 1
            
            

            # later on ... randomly adjust colours
            yield temp_chip, ((temp_mask > 0).astype(float)[..., np.newaxis], 
                              (temp_edge > 0).astype(float)[..., np.newaxis])
            break

def test_chips(imgs, mask,
               edge=None,
               padding=200,
               input_size=380,
               output_size=196):
    img_chips = []
    mask_chips = []
    if edge is not None:
        edge_chips = []

    center_offset = padding + (output_size / 2)
    for i, _ in enumerate(imgs):
        for x in np.arange(center_offset, imgs[i].shape[0] - input_size / 2, output_size):
            for y in np.arange(center_offset, imgs[i].shape[1] - input_size / 2, output_size):
                chip_x_l = int(x - (input_size / 2))
                chip_x_r = int(x + (input_size / 2))
                chip_y_l = int(y - (input_size / 2))
                chip_y_r = int(y + (input_size / 2))

                mask_x_l = int(x - (output_size / 2))
                mask_x_r = int(x + (output_size / 2))
                mask_y_l = int(y - (output_size / 2))
                mask_y_r = int(y + (output_size / 2))

                temp_chip = imgs[i][chip_x_l:chip_x_r, chip_y_l:chip_y_r]
                temp_mask = mask[i][mask_x_l:mask_x_r, mask_y_l:mask_y_r]
                if edge is not None:
                    temp_edge = edge[i][mask_x_l:mask_x_r, mask_y_l:mask_y_r]

                temp_chip = temp_chip.astype(np.float32) * 2
                temp_chip /= 255
                temp_chip -= 1

                img_chips += [temp_chip]
                mask_chips += [(temp_mask > 0).astype(float)[..., np.newaxis]]
                if edge is not None:
                    edge_chips += [(temp_edge > 0).astype(float)[..., np.newaxis]]

    img_chips = np.array(img_chips)
    mask_chips = np.array(mask_chips)
    if edge is not None:
        edge_chips = np.array(edge_chips)

    if edge is not None:
        return img_chips, mask_chips, edge_chips

    return img_chips, mask_chips

def test_chips3(imgs, mask,
               edge=None,
               padding=200,
               input_size=380,
               output_size=196):
    img_chips = []
    mask_chips = []
    if edge is not None:
        edge_chips = []

    center_offset = padding + (output_size / 2)
    for i, _ in enumerate(imgs):
        for x in np.arange(center_offset, imgs[i].shape[0] - input_size / 2, output_size):
            for y in np.arange(center_offset, imgs[i].shape[1] - input_size / 2, output_size):
                chip_x_l = int(x - (input_size / 2))
                chip_x_r = int(x + (input_size / 2))
                chip_y_l = int(y - (input_size / 2))
                chip_y_r = int(y + (input_size / 2))

                mask_x_l = int(x - (output_size / 2))
                mask_x_r = int(x + (output_size / 2))
                mask_y_l = int(y - (output_size / 2))
                mask_y_r = int(y + (output_size / 2))

                temp_chip = imgs[i][chip_x_l:chip_x_r, chip_y_l:chip_y_r]
                temp_mask = mask[i][mask_x_l:mask_x_r, mask_y_l:mask_y_r]
                if edge is not None:
                    temp_edge = edge[i][mask_x_l:mask_x_r, mask_y_l:mask_y_r]
                    
                    
                if temp_mask.shape != (100, 100, 1) and (temp_mask.shape != (100, 100)):
                    print("test_chips:")
                    print(f"img_chip shape : {temp_chip.shape}")
                    print(f"mask_chip shape : {temp_mask.shape}")
                    print(f"edge_chip shape : {temp_edge.shape}")
                    return temp_chip, temp_mask, temp_edge

                temp_chip = temp_chip.astype(np.float32) * 2
                temp_chip /= 255
                temp_chip -= 1

                img_chips += [temp_chip]
                mask_chips += [(temp_mask > 0).astype(float)[..., np.newaxis]]
                if edge is not None:
                    edge_chips += [(temp_edge > 0).astype(float)[..., np.newaxis]]

    img_chips = np.array(img_chips)
    mask_chips = np.array(mask_chips)
    if edge is not None:
        edge_chips = np.array(edge_chips)
    print("test_chips:")
    print(f"img_chips shape : {img_chips.shape}")
    print(f"mask_chips shape : {mask_chips.shape}")
    print(f"edge_chips shape : {edge_chips.shape}")

    if edge is not None:
        return img_chips, mask_chips, edge_chips

    return img_chips, mask_chips

def noisy(noise_typ,image, output_type = np.uint8):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy.astype(output_type)
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out.astype(output_type)
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy.astype(output_type)
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy.astype(output_type)
    
def remove_empty_images(image_files, mask_files, edge_files=None, keep_prob = 0, overwrite = False):
    
    new_image_files = []
    new_mask_files = []
    if edge_files is not None:
        new_edge_files = []

    #check the existance of dictionary of empty files
    dictPath = '/'.join(mask_files[0].split('/')[:-2]) + '/empty_files_dict.json'
    
    
    dictExists = os.path.exists(dictPath)
    
    if dictExists:
        # Opening JSON file
        with open(dictPath) as fp:
            # returns JSON object as
            # a dictionary
            empty_files_dict = json.load(fp)
        
        #integrity check
        if len(empty_files_dict) != len(mask_files):
            dictExists = False
            empty_files_dict = {}
    else:
        empty_files_dict = {}
    
    
    
    
    for i in tqdm(range(len(image_files))):
        
        keep = np.random.choice([True, False], p=[keep_prob, 1 - keep_prob])
        
        if not dictExists:
            image = cv2.imread(image_files[i])
        
            empty_files_dict[image_files[i]] = (len(np.unique(image)) < 5)
            
            if (len(np.unique(image)) < 5) and (not keep):
                #print(f"deleting image {image_files[i]} where the max is {image.max()} unique is {len(np.unique(image))} keep {keep}")
                del image
                continue
            else:
                #print(f"keeping image {image_files[i]} where the max is {image.max()} unique is {len(np.unique(image))} keep {keep}")
                del image
        else:
            if (empty_files_dict[image_files[i]] == True) and (not keep):
                continue
        
        new_image_files += [image_files[i]]
        new_mask_files += [mask_files[i]]
        if edge_files is not None:
            new_edge_files += [edge_files[i]]
            
    if not dictExists:
        with open(dictPath, "w+") as fp:
            json.dump(empty_files_dict,fp)
    
    if edge_files is not None:
        return new_image_files, new_mask_files, new_edge_files
    
    return new_image_files, new_mask_files  

def slice_images(imgs,
               padding=200,
               input_size=188,
               output_size=100):
    img_chips = []
    img_sizes = []

    center_offset = padding + (output_size / 2)
    for i, _ in enumerate(imgs):
        img_sizes += [(len(np.arange(center_offset, imgs[i].shape[0] - input_size / 2, output_size)), len(np.arange(center_offset, imgs[i].shape[1] - input_size / 2, output_size)))]
        for x in np.arange(center_offset, imgs[i].shape[0] - input_size / 2, output_size):
            for y in np.arange(center_offset, imgs[i].shape[1] - input_size / 2, output_size):
                chip_x_l = int(x - (input_size / 2))
                chip_x_r = int(x + (input_size / 2))
                chip_y_l = int(y - (input_size / 2))
                chip_y_r = int(y + (input_size / 2))


                temp_chip = imgs[i][chip_x_l:chip_x_r, chip_y_l:chip_y_r]
               

                temp_chip = temp_chip.astype(np.float32) * 2
                temp_chip /= 255
                temp_chip -= 1

                img_chips += [temp_chip]
                
    img_chips = np.array(img_chips)


    return img_chips, img_sizes


def concat_slices(im_slices_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_slices_2d[:,:,:,:,:]])

        
        
def plot_ime(imgs,
             masks,
             edges,
             lines = 2,
             columns = 2,
             figSize = (15,10),
             threshold = 0,
             prefix = "",
             max_plots = 20):
    
    if max_plots > len(imgs):
        max = len(imgs)
    else:
        max = max_plots
    
    for i in np.arange(max):
        image = imgs[i]
        mask = masks[i]
        edge = edges[i]        
        
        fig = plt.figure(figsize=figSize, dpi=80)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        fig_num = 1
        ax = fig.add_subplot(lines, columns, fig_num)
        ax.set_title(f"{prefix} image {i}")
        ax.imshow(image)
        fig_num = fig_num + 1
        ax = fig.add_subplot(lines, columns, fig_num)
        ax.set_title(f"{prefix} mask {i}")
        ax.imshow(np.squeeze(mask))
        fig_num = fig_num + 1
        ax = fig.add_subplot(lines, columns, fig_num)
        ax.set_title(f"{prefix} edge mask {i}")
        ax.imshow(np.squeeze(edge))
        fig_num = fig_num + 1
        ax = fig.add_subplot(lines, columns, fig_num)
        ax.set_title(f"{prefix} substraction {i}")
        ax.imshow(np.squeeze((mask - edge) > threshold))
        

#function to plot an array of images of shape (width, height, n_images)       
def picshow(img, title):
    num = img.shape[2]
    imgs_per_line = 4
    ax = num//imgs_per_line + 1
    ay = imgs_per_line
    fig =plt.figure(figsize=(30, num//imgs_per_line * 10))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    for i in range(1,num + 1):
        sub = fig.add_subplot(ax,ay,i)
        sub.set_title(f"{title} {i}")
        sub.imshow(img[:-1,:-1,i - 1])
    plt.show()
    
    
def scaleBetween(data ,scaledMin, scaledMax):
    max = data.max()
    min = data.min()
    res = (scaledMax-scaledMin)*(data-min)/(max-min)+scaledMin
    return res
  
def chunks(l, n = None):
    if n == None:
        n = len(l) // 10
        
    n = max(1, n)
    return [l[i:i+n] for i in range(0, len(l), n)]

def showImg(img, title="image", figSize=(15, 20), dpi=80):
    fig = plt.figure(figsize=figSize, dpi=dpi)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.set_title(title)
    ax.imshow(img)
    
#this function takes a gray scale image and filters non connected objects by size
def surfaceFilter(image, min_size = None, max_size = None, colorize = False, gray = False):
    img = image.copy()
    
    unique_num = 2147483647
    
    #showImg(img, "input")
    
    ret, labels = cv2.connectedComponents(img)
    
    #(ret, labels, values, centroid) = cv2.connectedComponentsWithStats(img)
    
    #print(labels.dtype)
    
    
    
    label_codes = np.unique(labels)
    
    # print(ret)
    
    # print(np.unique(labels))
    
    #labels = np.asarray(labels) + 1
    
    # print(np.unique(labels))
    
    # showImg(labels, "lables")
    
    result_image = labels
    
    if unique_num  in result_image:
        print(f"error the image contains the null number {unique_num}")
    
    i = 0
    background_index = 0
    max = 0
    for label in label_codes:
        count = (labels == label).sum()
        
        #find the background index
        if count > max:
            max = count
            background_index = i
        
        
        if min_size is not None and (count < min_size):
            result_image[labels == label] = unique_num
            #print(f"min: removing {count} because is < than {min_size}")
        # else:
        #     if min_size is not None:
        #         print(f"min: {count} is > than {min_size}")
        if max_size is not None and (count > max_size):
            result_image[labels == label] = unique_num
        #     print(f"max: removing {count} because is > than {max_size}")
        # else:
        #     if max_size is not None:
        #         print(f"max: {count} is < than {max_size}")
        
        i = i + 1
    
    
    #print(f"background index is {background_index} with count = {max}")
    result_image[result_image == unique_num] = label_codes[background_index]
            
    if colorize:
        result_image = colorize_unique(result_image)
        
        if gray:
            result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)
    
    #showImg(result_image, "output")
    return result_image

def colorize_unique(image):
    # Map component labels to hue val
    label_hue = np.uint8(179 * image / np.max(image))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0
    return labeled_img


