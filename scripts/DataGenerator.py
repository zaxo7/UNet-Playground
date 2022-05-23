from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import cv2


import data, model


class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, image_files, mask_files, edge_files=None,
                 batch_size=8,
                 input_size=(188, 188, 3),
                 augment = True,
                 shuffle = False,
                 skip_empty = False,
                 keep_empty_prob = 0.05,
                 slice = False,
                 min_resize = 30):
        
        self.image_files = image_files
        self.mask_files = mask_files
        
        self.edge_files = edge_files

        self.batch_size = batch_size
        self.input_size = input_size
        
        self.augment = augment
        self.shuffle = shuffle
        self.skip_empty = skip_empty
        self.keep_empty_prob = keep_empty_prob
        self.min_resize = min_resize        
        self.slice = slice
        
        
        
        
        if skip_empty:
            print(f"skipping empty images with keep probability of {self.keep_empty_prob}\nbefore skip we have {len(self.image_files)} images")
            if self.edge_files is not None:
                self.image_files, self.mask_files, self.edge_files = data.remove_empty_images(self.image_files, self.mask_files, self.edge_files, keep_prob = self.keep_empty_prob)
            else:
                self.image_files, self.mask_files = data.remove_empty_images(self.image_files, self.mask_files, keep_prob = self.keep_empty_prob)
            print(f"after skip we have {len(self.image_files)} images")
        
        self.n = len(self.image_files)
    
    def on_epoch_end(self):
        if self.edge_files is not None:
            self.image_files, self.mask_files, self.edge_files = shuffle(self.image_files, self.mask_files, self.edge_files)
        else:
            self.image_files, self.mask_files = shuffle(self.image_files, self.mask_files)
            
    
    def __load_data(self, image_files, mask_files, edge_files=None):
        return data.load_data3(image_files, mask_files, edge_files, preprocess = False)
        
    def __normalize(self, images, masks, edges = None):
        
        for i in np.arange(len(images)):
            image = images[i]
            mask = masks[i]
            if edges is not None:
                edge = edges[i]
            
            
            # rescale the image // normalisation to [-1,1] range
            image = image.astype(np.float32) * 2
            image /= 255
            image -= 1
            
            images[i] = image
            masks[i] = (mask > 0).astype(np.float32)#[..., np.newaxis]
            if edges is not None:
                edges[i] = (edge > 0).astype(np.float32)#[..., np.newaxis]
        
        if edges is not None:
            return images, masks, edges
        
        return images, masks
        
    
    def __augment(self, images, masks, edges = None):
        
        # Select which type of cell to return // sometimes keep empty images
        #chip_type = np.random.choice([True, False])
        chip_type = True
        limit = len(images)
        i = 0
        while i < limit:
            
            image = images[i]
            mask = masks[i]
            if edges is not None:
                edge = edges[i]
                    
            # randomly rotate
            rot = np.random.randint(4)
            image = np.rot90(image, k=rot, axes=(0, 1))
            mask = np.rot90(mask, k=rot, axes=(0, 1))
            if edges is not None:
                edge = np.rot90(edge, k=rot, axes=(0, 1))

            # randomly flip
            if np.random.random() > 0.5:
                image = np.flip(image, axis=1)
                mask = np.flip(mask, axis=1)
                if edges is not None:
                    edge = np.flip(edge, axis=1)
                
            #add some noise to image
            noise_type = np.random.choice(['gauss', 'poisson', 's&p', 'speckle'])
            image = data.noisy(noise_type, image)
            
            #reduce image quality
            
            min_scale = np.random.randint(30 // 2, 188 // 2) * 2
            
            image = cv2.resize(image, (min_scale, min_scale), interpolation= cv2.INTER_LINEAR)
            image = cv2.resize(image, (188,188), interpolation= cv2.INTER_LINEAR)
            
            # randomly luminosity augment
            image = data.aug_img(image)
            
            #TODO add blur and noise (maybe with quad tree we can do some noise)
            
            images[i] = image
            masks[i] = mask
            if edges is not None:
                edges[i] = edge
            
            
            i = i + 1

            
        
        if self.edge_files is not None:
            return images, masks, edges
        else:
            return images, masks
                
            
    
    def __getitem__(self, index):
        
            
        
        #we need to slice the images
        if self.slice:
            image_files = self.image_files
            mask_files = self.mask_files
            if self.edge_files is not None:
                edge_files = self.edge_files
            
            if self.edge_files is not None:
                image_files, mask_files, edge_files = shuffle(image_files, mask_files, edge_files)
            else:
                image_files, mask_files = shuffle(image_files, mask_files)
            images = []
            masks = []
            edges = []
            
            #we keep slicing random images untill we get the batch size
            i = 0
            while(len(images) < self.batch_size):
            
                if self.edge_files is not None:
                    image, mask, edge = model.generate_test_dataset3([image_files[i]], [mask_files[i]], [edge_files[i]])
                else:
                    image, mask = model.generate_test_dataset3([image_files[i]], [mask_files[i]])
                
                if i == 0:
                    images = image
                    masks = mask
                    if self.edge_files is not None:
                        edges = edge
                else:
                    images += image
                    masks += mask
                    if self.edge_files is not None:
                        edges += edge

            #take n=batchsize random elements from the lists
            if self.edge_files is not None:
                images, masks, edges = shuffle(images, masks, edges)
            else:
                images, masks = shuffle(images, masks)
                    
            images = images[0:self.batch_size]
            masks = masks[0:self.batch_size]
            if self.edge_files is not None:
                edges = edges[0:self.batch_size]
        else:
            image_files = self.image_files[index * self.batch_size:(index + 1) * self.batch_size]
            mask_files = self.mask_files[index * self.batch_size:(index + 1) * self.batch_size]
            if self.edge_files is not None:
                edge_files = self.edge_files[index * self.batch_size:(index + 1) * self.batch_size]
            
            if self.edge_files is not None:
                images, masks, edges = self.__load_data(image_files, mask_files, edge_files)
            else:
                images, masks = self.__load_data(image_files, mask_files)
                
                
        
        if self.augment:
            if self.edge_files is not None:
                images, masks, edges = self.__augment(images, masks, edges)
            else:
                images, masks = self.__augment(images, masks)
            
        if self.edge_files is not None:
            images, masks, edges = self.__normalize(images, masks, edges)
        else:
            images, masks = self.__normalize(images, masks)
            
              
            
        images = np.asarray(images)
        
        if not self.slice:
            masks = np.asarray(masks).astype(np.float32)[..., np.newaxis]
            if self.edge_files is not None:
                edges = np.asarray(edges).astype(np.float32)[..., np.newaxis]
        
        
        if self.edge_files is not None:
            print(f"loading batch number {index} which has shape image : {images.shape} mask : {masks.shape} edge : {edges.shape}")
            return images, (masks, edges)
    
        print(f"loading batch number {index} which has shape image : {images.shape} mask : {masks.shape}")
        return images, (masks)
    
    def __len__(self):
        return self.n // self.batch_size
