import os
import numpy as np
import cv2
import albumentations
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


# classes for data loading and preprocessing
class Dataset:
    """ Dataset Classs. Read images, apply augmentation and preprocessing transformations.

    Args:
        base_dir (str): path to base folder
        classes (list): classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. normalization, shape manipulation, etc.)
        size (int): size needed for image
        db (str): name of database/dataset to use
        train (bool): Load training or testing (valid) portion
    """

    CLASSES = ['built-up', 'background']
    supported_dbs = ['TeamUrbanTech', 'SpaceNet']
    def __init__(
            self,
            base_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
            size=1024,
            db='TeamUrbanTech',
            train=True
    ):
        # assert self.train is not None, "Pass the train argument data"
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.size = size
        self.db = db
        self.train = train

        if self.db == self.supported_dbs[1]:
            images_dir = os.path.join(base_dir, 'images')
            image_ids = np.sort(os.listdir(images_dir))
            self.image_paths = [os.path.join(images_dir, image_id) for image_id in image_ids]
            masks_dir = os.path.join(base_dir, 'masks')
            self.mask_paths = [os.path.join(masks_dir, image_id.replace(".npy", "_mask.npy")) for image_id in image_ids]
            length = len(self.image_paths)
            if self.train:
                self.image_paths = self.image_paths[:int(0.8*length)]
                self.mask_paths = self.mask_paths[:int(0.8*length)]
            else:
                self.image_paths = self.image_paths[int(0.8 * length):]
                self.mask_paths = self.mask_paths[int(0.8 * length):]
        else:
            if self.train:
                images_dir = os.path.join(base_dir, 'images')
                image_ids = np.sort(os.listdir(images_dir))
                self.image_paths = [os.path.join(images_dir, image_id) for image_id in image_ids]
                masks_dir = os.path.join(base_dir, 'masks')
                mask_ids = np.sort(os.listdir(masks_dir))
                self.mask_paths = [os.path.join(masks_dir, mask_id) for mask_id in mask_ids]
            else:
                images_dir = os.path.join(base_dir, 'test_images')
                image_ids = np.sort(os.listdir(images_dir))
                self.image_paths = [os.path.join(images_dir, image_id) for image_id in image_ids]
                masks_dir = os.path.join(base_dir, 'test_masks')
                mask_ids = np.sort(os.listdir(masks_dir))
                self.mask_paths = [os.path.join(masks_dir, mask_id) for mask_id in mask_ids]

    def __getitem__(self, i):
        assert self.db in self.supported_dbs, "The provided database: {} is not supported".format(self.db)

        # read data
        if self.db==self.supported_dbs[0]: #Our data
            image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.mask_paths[i])
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

            # convert RGB mask to index
            one_hot_map = []
            colors = [
                (164, 113, 88),  # Built-up
                (255, 255, 255)  # background
            ]
            for color in colors:
                class_map = np.all(np.equal(mask, color), axis=-1)
                one_hot_map.append(class_map)

            one_hot_map = np.stack(one_hot_map, axis=-1)
            one_hot_map = one_hot_map.astype('float32')

            mask = np.argmax(one_hot_map, axis=-1)

            # Label conflicting edges as last class(background)
            min_mask = np.argmin(one_hot_map, axis=-1)
            mask[(mask == min_mask)] = (len(colors)) - 1

            # extract certain classes from mask (e.g. cars)
            masks = [(mask == v) for v in self.class_values]
            mask = np.stack(masks, axis=-1).astype('float')

            # add background if mask is not binary
            if mask.shape[-1] != 1:
                background = 1 - mask.sum(axis=-1, keepdims=True)
                mask = np.concatenate((mask, background), axis=-1)
        else:
            image = np.load(os.path.join(self.image_paths[i]))
            mask = np.load(os.path.join(self.mask_paths[i]))
            mask = (mask > 0).astype(np.int32)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask.astype('uint8'), (self.size, self.size), interpolation=cv2.INTER_AREA)

        # print("Mask Shape after process 4: {}".format(mask.shape))
        image = np.transpose(image, (2, 0, 1)).astype('float32')

        return image, mask, self.image_paths[i]

    def __len__(self):
        return len(self.image_paths)