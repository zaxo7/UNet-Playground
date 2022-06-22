import data, model

from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from sklearn.metrics import r2_score
from scipy import ndimage
import numpy as np
import imutils

import cv2

from math import pi

from scripts.data import showImg

#for RBC

#for WBC
#local_max_min_dist = 50
#min_filter_size = 1000
#thresh =  binary 127
def Watershed_Count(_image, _mask, plot = False, plot_res = True, return_image = False, min_filter_size = 400, max_filter_size = None, threshold_type = "binary", local_max_min_dist = 50):
    
    mask = (_mask.copy() * 255.0).astype(np.uint8)
    
    mask_thresh_binary = None
    if threshold_type == "binary":
        mask_thresh_binary= ((mask > 127) * 255.0).astype(np.uint8)
        
        if plot:
            data.showImg(mask_thresh_binary, title="binary thresh  127")
    elif threshold_type == "otsu":
        ret2, mask_gray_otsu = cv2.threshold((mask * 255.0).astype(np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        mask_thresh_binary = mask_gray_otsu
    
        if plot:
            data.showImg(mask_gray_otsu, title="otsu")
    else:
        print("error unknown thresh")
        return
    
    
    
    
    if plot:
        data.showImg(mask, title="original mask")
    
    
    mask_clean_gray = data.surfaceFilter(mask_thresh_binary, min_size = min_filter_size, max_size= max_filter_size, colorize = False, gray=False)
    

    if plot:
        data.showImg(mask_clean_gray, title="clean mask gray scale")


    mask_clean_binary = ((mask_clean_gray > 0) * 255.0).astype(np.uint8)
    
    if plot:
        data.showImg(mask_clean_binary, title="clean mask binary")

    D = ndimage.distance_transform_edt(mask_clean_binary)


    if plot:
        data.showImg(D, title="euclidian Distance Transform")


    localMax = peak_local_max(D, 
                            indices=False, 
                            min_distance=local_max_min_dist, 
                            labels=mask_clean_binary)

    if plot:
        data.showImg(localMax, "local maxiama")

    # perform a connected component analysis on the local peaks to label each local maxima with a num,
    # using 8-connectivity, then appy the Watershed algorithm we can also use connected component labeling
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    
    #(nbcells, out_image, values, centroid) = cv2.connectedComponentsWithStats(localMax.astype(np.uint8))
    
    #print(f"found {nbcells} with localMax")

    if plot:
        #same image as local maxima
        data.showImg(markers, title="local maxiama labled markers")

    labels = watershed(-D, markers, mask=mask_clean_gray)

    #print("[INFO] {} WBC's found with Watershed".format(len(np.unique(labels)) - 1))
    
    cells =  len(np.unique(labels)) - 1

    if plot:
        data.showImg(labels, title="watershed result")

        data.showImg(data.colorize_unique(labels))

    #loop over the unique labels returned by the Watershed
    # algorithm
    image_orig = _image.copy()
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(mask.shape, dtype="uint8")
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        cv2.circle(image_orig, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(image_orig, "#{}".format(label), (int(x) - 10, int(y)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    #if plot:
        # show the output image
    if plot_res:    
        data.showImg(image_orig, f"Watershed Final Output with {cells} cells")
        
    del mask
    del mask_thresh_binary
    del mask_clean_gray
    del mask_clean_binary
    del D
    del localMax
    del markers
    del labels
        
    if return_image == True:
        return (cells, image_orig)
    
    del image_orig
    return cells




def CCL_Count(_image, _mask, plot = False, plot_res = True, return_image = False, min_filter_size = 400, max_filter_size = None, threshold_type = "binary"):
    
    mask = (_mask.copy() * 255.0).astype(np.uint8)
    
    mask_thresh_binary = None
    if threshold_type == "binary":
        mask_thresh_binary= ((mask > 127) * 255.0).astype(np.uint8)
        
        if plot:
            data.showImg(mask_thresh_binary, title="binary thresh  127")
    elif threshold_type == "otsu":
        ret2, mask_gray_otsu = cv2.threshold((mask * 255.0).astype(np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        mask_thresh_binary = mask_gray_otsu
    
        if plot:
            data.showImg(mask_gray_otsu, title="otsu")
    else:
        print("error unknown thresh")
        return
    
    if plot:
        data.showImg(mask, title="original mask")
    
    
    mask_clean_gray = data.surfaceFilter(mask_thresh_binary, min_size = min_filter_size, max_size= max_filter_size, colorize = True, gray=True)
    

    if plot:
        data.showImg(mask_clean_gray, title="clean mask gray scale")


    mask_clean_binary = ((mask_clean_gray > 0) * 255.0).astype(np.uint8)
    
    if plot:
        data.showImg(mask_clean_binary, title="clean mask binary")
        
        
    (cells, labels, values, centroid) = cv2.connectedComponentsWithStats(mask_clean_binary)
    
    
    #loop over the unique labels returned by the Watershed
    # algorithm
    image_orig = _image.copy()
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(mask.shape, dtype="uint8")
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        cv2.circle(image_orig, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(image_orig, "#{}".format(label), (int(x) - 10, int(y)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    #if plot:
        # show the output image
    if plot_res:    
        data.showImg(image_orig, f"Connected Component Labeling Final Output with {cells} cells")
        
    del mask
    del mask_thresh_binary
    del mask_clean_gray
    del mask_clean_binary
    del labels
    
    if return_image == True:
        return(cells, image_orig)
    
    del image_orig
    
    
    return cells


def CHT_Count(_image, _mask, plot = False, plot_res = True, return_image = False, min_filter_size = 400, max_filter_size = None, min_radius = 40, max_radius = 100, min_dist = 50, param1 = 1, param2 = 1, threshold_type = "binary", threshold = 80):
    
    mask = (_mask.copy() * 255.0).astype(np.uint8)
    
    mask_thresh_binary = None
    if threshold_type == "binary":
        mask_thresh_binary= ((mask > 127) * 255.0).astype(np.uint8)
        
        if plot:
            data.showImg(mask_thresh_binary, title="binary thresh  127")
    elif threshold_type == "otsu":
        ret2, mask_gray_otsu = cv2.threshold((mask * 255.0).astype(np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        mask_thresh_binary = mask_gray_otsu
    
        if plot:
            data.showImg(mask_gray_otsu, title="otsu")
    else:
        print("error unknown thresh")
        return
    
    if plot:
        data.showImg(mask, title="original mask")
    
    
    mask_clean_gray = data.surfaceFilter(mask_thresh_binary, min_size = min_filter_size, max_size= max_filter_size, colorize = True, gray=True)
    

    if plot:
        data.showImg(mask_clean_gray, title="clean mask gray scale")


    mask_clean_binary = ((mask_clean_gray > 0) * 255.0).astype(np.uint8)
    
    if plot:
        data.showImg(mask_clean_binary, title="clean mask binary")
    
    
    
    circles = cv2.HoughCircles(mask_clean_binary, cv2.HOUGH_GRADIENT, 1, minDist= min_dist,
                          param1=param1, param2=param2,
                          minRadius=min_radius, maxRadius=max_radius)
    if circles is not None:
        circles = (circles[0]).astype(np.uint)
    else:
        circles = []

    #print(f"found {len(circles)} circles before cleaning")
    
    
    image_circ = _image.copy()
    for i in np.arange(len(circles)):
        cv2.circle(image_circ, (circles[i,0], circles[i,1]), circles[i,2], color=(0,0,255), thickness=3)
        #Draw Center (red)
        #cv2.circle(image_circ, (circles[i,0], circles[i,1]), 3, color=(0,255,0), thickness=4)
    if plot:   
        data.showImg(image_circ)
    
    circles = CL(_image, mask_clean_binary, circles=circles, threshold=threshold)
    
    #print(circles)
    
    #print(f"found {len(circles)} circles after cleaning")

    image_circ = _image.copy()
    for i in np.arange(len(circles)):
        cv2.circle(image_circ, (circles[i][0], circles[i][1]), circles[i][2], color=(0,0,255), thickness=3)
        #Draw Center (red)
        #cv2.circle(image_circ, (circles[i][0], circles[i][1]), 3, color=(0,255,0), thickness=4)
    if plot_res:    
        data.showImg(image_circ, f"Circle Hough Transform Final Output with {len(circles)} cells")
    
    if return_image == True:
        return (len(circles), image_circ)
    
    return len(circles)
    
    
        
    


def CL(_image, _mask, circles, threshold = 80):
    image = cv2.resize(_image, (_mask.shape[1], _mask.shape[0])) 
    mask = _mask.copy()

    # data.showImg(image, "input image")
    # data.showImg(mask, "input mask")
    
    mask_bin = mask
        
    #data.showImg(mask_bin)
    
    new_circles = []
    
    #calculate the area of overlap between each circle and the mask
    for circle in circles:
        tmp_image = np.zeros_like(image)
        cv2.circle(tmp_image, (circle[0], circle[1]), circle[2], color=(255,255,255), thickness=-1)
        tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_BGR2GRAY)
        # data.showImg(tmp_image)
        
        # print(np.unique(tmp_image))
        # print(np.unique(mask_bin))
        
        tmp_image_rgb = cv2.cvtColor(tmp_image, cv2.COLOR_GRAY2BGR).astype(np.uint8)
        mask_bin_rgb = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2BGR).astype(np.uint8)
        
        # print(tmp_image_rgb.shape)
        # print(mask_bin_rgb.shape)
        
        overlap = cv2.bitwise_or(mask_bin_rgb, mask_bin_rgb, mask=tmp_image)
        overlap = cv2.cvtColor(overlap, cv2.COLOR_BGR2GRAY)
        #print(f"{overlap} pixels of mask overlaped with circle")
        
        
        #calculate circle surface
        circle_surface  = pi * circle[2]**2
        # calculate the surface of intercection        
        overlap_surface = (overlap > 0).sum()
        
        if overlap_surface > circle_surface:
            print(f"error {circle_surface} < {overlap_surface}")
            data.showImg(overlap)
        
        
        overlap_percentage = (overlap_surface) * 100 / circle_surface
        
        if overlap_percentage > threshold:
            new_circles += [circle.tolist()]
        # else:
        #     print(f"{overlap_percentage} <= {threshold}")

    
    return new_circles


def WBC_Count(images_path, trained_model, _masks=None):
    
    ########################
    
    manual_count_file = open("ALL-IDB1_WBC_Count.txt", "r")

    manual_count_file_lines = manual_count_file.readlines()
    manual_count_file_lines = ''.join(manual_count_file_lines).split("\n")

    manual_counts_f = []
    manual_counts = []

    for line in manual_count_file_lines:
        split = line.split(" ")
        manual_counts_f += [(split[0], split[1])]
        manual_counts += [split[1]]
        
        
    manual_counts = np.asarray(manual_counts).astype(np.int32)
        
    ########################
  
    min_filter_size = 1000
    overlap_threshold = 60
  
    i = 0
    
    watershed_counts = []
    ccl_counts = []
    cht_counts = []
    
    real_counts = []
    
    for image in images_path:
        if _masks is None:
            images, masks = model.predictFullImage(trained_model,
                                data.load_data_na([image], RGB=True, clahe=True),
                                padding=100,
                                input_size=188,
                                output_size=100,
                                normalize_output = False,
                                edge=False)
            del images
        else:
            masks = _masks
        
        original_image = data.load_data_na([image], RGB=True, preprocess = True, padding=100)
        
        print(f"counting WBC's in image {image}")
        
        data.showImg(original_image[0], f"{image}")
        data.showImg(masks[0], f"{image}")
        
        watershed_count = Watershed_Count(original_image[0], masks[0], plot = False, min_filter_size=min_filter_size, threshold_type="binary")
        watershed_counts += [watershed_count]
        
        ccl_count = CCL_Count(original_image[0], masks[0], plot=False, min_filter_size=min_filter_size)
        ccl_counts += [ccl_count]      
        
        cht_count = CHT_Count(original_image[0], masks[0], plot=False, min_filter_size=min_filter_size, param1 = 500, param2 = 7.6, min_dist=80, threshold = overlap_threshold)
        cht_counts += [cht_count]
        
        real_counts += [manual_counts[i]]
        
        
        print(f"watershed = {watershed_count}")
        print(f"ccl = {ccl_count}")
        print(f"circle hough transform = {cht_count}")
        
        
        watershed_accuracy = (1 - (abs(manual_counts[i] - watershed_count)) / manual_counts[i]) * 100
        # watershed_R2 = r2_score([manual_counts[i]], [watershed_count])
        
        ccl_accuracy = (1 - (abs(manual_counts[i] - ccl_count)) / manual_counts[i]) * 100
        # ccl_R2 = r2_score([manual_counts[i]], [ccl_count])


        cht_accuracy = (1 - (abs(manual_counts[i] - cht_count)) / manual_counts[i]) * 100
        # cht_R2 = r2_score([manual_counts[i]], [cht_count])

        
        log(image, manual_counts[i], watershed_count, ccl_count, cht_count, watershed_accuracy, ccl_accuracy, cht_accuracy)
        
        del original_image
        i += 1
    
    #last line of log file we put the total accuracy and r2 score
    watershed_accuracy = (1 - (np.abs(np.subtract(real_counts, watershed_counts))) / real_counts) * 100
    # watershed_R2 = r2_score(real_counts, watershed_counts)
    
    ccl_accuracy = (1 - (np.abs(np.subtract(real_counts, ccl_counts))) / real_counts) * 100
    # ccl_R2 = r2_score(real_counts, ccl_counts)
    
    cht_accuracy = (1 - (np.abs(np.subtract(real_counts, cht_counts))) / real_counts) * 100
    # cht_R2 = r2_score(real_counts, cht_counts)
    
    log("Total", -1, -1, -1, -1, watershed_accuracy, ccl_accuracy, cht_accuracy)
      
      
#def log(image_name, real_count, watershed_count, ccl_count, cht_count, watershed_R2, watershed_accuracy, ccl_R2, ccl_accuracy, cht_R2, cht_accuracy, log_file = "results.txt"):
def log(image_name, real_count, watershed_count, ccl_count, cht_count, watershed_accuracy, ccl_accuracy, cht_accuracy, log_file = "results.txt"):
    separator = '\t'
    watershed_accuracy = watershed_accuracy.mean()
    ccl_accuracy = ccl_accuracy.mean()
    cht_accuracy = cht_accuracy.mean()
    line = f"{image_name}{separator}{real_count}{separator}{watershed_count}{separator}{ccl_count}{separator}{cht_count}{separator}{watershed_accuracy:.2f}{separator}{ccl_accuracy:.2f}{separator}{cht_accuracy:.2f}{separator}"
    
    with open(log_file, "a+") as file:
        file.write(line + "\n")
