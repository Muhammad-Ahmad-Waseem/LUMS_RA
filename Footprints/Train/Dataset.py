'''
Dataset. Read images, apply augmentation and preprocessing transformations.
Specified for building footprints (3 class code). Assumes that building mask
and contour mask have same color while masks and contours are saved in diff-
erent folders.
Args:
    images_dir (str): path to images folder
    masks_dir (str): path to building masks folder
    contr_dir (str): path to building outlines/footprints/contours mask folder
    augmentation (albumentations.Compose): data transfromation pipeline
        (e.g. flip, scale, etc.)
    preprocessing (albumentations.Compose): data preprocessing
        (e.g. normalization, shape manipulation, etc.)
'''

import os
import numpy as np
import cv2

class Dataset:
    CLASSES = ['built-up','background']
    colors = [
        (164, 113, 88), # Built-up
        (255, 255, 255) # background
        ]
    mode_choices = ['multiclass', 'multilabel']
    def __init__(
            self, 
            images_dir, 
            masks_dir,
            contr_dir,
            augmentation=None, 
            preprocessing=None,
            mode = 'multiclass'
    ):
        assert (mode in self.mode_choices), "mode type not found. Choose one of these: {}".format(self.mode_choices)
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in np.sort(os.listdir(images_dir))]
        self.mask_paths = [os.path.join(masks_dir, mask_id) for mask_id in np.sort(os.listdir(masks_dir))]
        self.contr_paths = [os.path.join(contr_dir, contr_id) for contr_id in np.sort(os.listdir(contr_dir))]
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.mode = mode
    
    def __getitem__(self, i):
        # read data
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[i])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        contr = cv2.imread(self.contr_paths[i])
        contr = cv2.cvtColor(contr, cv2.COLOR_BGR2RGB)

        # convert RGB mask to index 
        one_hot_map = []
        contr_map = np.all(np.equal(contr, self.colors[0]), axis=-1)
        one_hot_map.append(contr_map)
        mask_map = np.all(np.equal(mask, self.colors[0]), axis=-1)
        one_hot_map.append(mask_map)
        back_map = np.all(np.equal(mask, self.colors[1]), axis=-1)
        one_hot_map.append(back_map)
        
        one_hot_map = np.stack(one_hot_map, axis=-1)
        one_hot_map = one_hot_map.astype('float32')
        
        labels = np.argmax(one_hot_map, axis=-1)

        if self.mode == self.mode_choices[1]:
            # extract certain classes from mask (e.g. cars)
            masks = [(labels == v) for v in range(3)]
            mask = np.stack(masks, axis=-1).astype('float')
        else:
            mask = labels

        #print(mask.shape)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        image = np.transpose(image, (2, 0, 1)).astype('float32')
        if self.mode == self.mode_choices[1]:
            mask = np.transpose(mask, (2, 0, 1)).astype('int64')

        return image,mask.astype('int64')#,self.image_ids[i]
        
    def __len__(self):
        return len(self.image_paths)