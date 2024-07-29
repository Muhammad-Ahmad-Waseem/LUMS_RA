import os
import argparse
#import tensorflow as tf 
#tf.enable_eager_execution()
#import tensorflow as tf
#import tensorflow.keras as keras
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt

import albumentations as A
import segmentation_models as sm
import segmentation_models_pytorch as smp

sm.set_framework('keras')
sm.framework()
#//////////////////////// ARGUMENT PARSER \\\\\\\\\\\\\\\\\\\\\\\\
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../Datasets/Segmentation_Data')
parser.add_argument('--output_dir', default='./Keras/trained_model/satellite')
parser.add_argument('--backbone', default='efficientnetb3')
parser.add_argument('-m',   '--mode', choices=['t', 'es', 'ct', 'eb', 'v', 'ph'], default='es', required='true')

#//////////////////////// DATA LADER and UTILITY FUNCTIONS \\\\\\\\\\\\\\\\\\\\\\\\

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
    
# helper function for data visualization    
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x
    
# helper function for creating RGB of ith mask
def MasktoRGB(mask,color):
    r = np.expand_dims((mask*color[0]),axis=-1)
    g = np.expand_dims((mask*color[1]),axis=-1)
    b = np.expand_dims((mask*color[2]),axis=-1)
    img = np.concatenate((r,g,b), axis=-1)
    return img.astype(int)


# classes for data loading and preprocessing
class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    #CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 'tree', 'signsymbol', 'fence', 'car', 'pedestrian', 'bicyclist', 'unlabelled']
    '''CLASSES = ['hospitals','commercial', 'institutional', 'mosques', 'residential', 'transport&util', 'underconst', 'vegetation', 'others','background']
    colors = [
        (253, 191, 111), # Hospitals
        (0, 255, 251), # Commercial
        (31, 120, 180), # Institutional
        (0, 255, 0), # Mosques
        (234, 113, 245), # Residential
        (244, 244, 40), # Transport&Util
        (255, 127, 0), # UnderConst
        (38, 137, 23), # Vegetation
        (0, 0, 0), # Others
        (255, 255, 255) # background
        ]'''
    CLASSES = ['built-up', 'underconst', 'vegetation','background']
    colors = [
        (164, 113, 88), # Built-up
        (255, 127, 0), # UnderConst
        (38, 137, 23), # Vegetation
        (255, 255, 255) # background
        ]
    '''
    CLASSES = ['background','grass']
    colors = [
        (0, 0, 0), # background
        (128, 0, 0) # grass
        ]
    CLASSES = ['commercial', 'residential','background']
    colors = [
        (0, 255, 251), # Commercial
        (234, 113, 245), # Residential
        (255, 255, 255) # background
        ]'''
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.image_ids = np.sort(os.listdir(images_dir))
        #print(self.image_ids)
        self.mask_ids = np.sort(os.listdir(masks_dir))
        #print(self.mask_ids)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.image_ids]
        self.masks_fps = [os.path.join(masks_dir, mask_id) for mask_id in self.mask_ids]
        
        # convert str names to class values on masks
        
        
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        #print(mask.shape)
        # convert RGB mask to index 
        one_hot_map = []
        
        for color in self.colors:
            class_map = np.all(np.equal(mask, color), axis=-1)
            #print(class_map[300:301,200:210])
            one_hot_map.append(class_map)
        
        one_hot_map = np.stack(one_hot_map, axis=-1)
        #one_hot_map = tf.cast(one_hot_map, tf.float32)
        one_hot_map = one_hot_map.astype('float32')
        
        mask = np.argmax(one_hot_map, axis=-1)
        
        #Label conflicting edges as last class(background)
        min_mask = np.argmin(one_hot_map, axis=-1)
        mask[(mask==min_mask)]=(len(self.colors))-1
        
        #mask = mask.numpy()
        #tf.print(mask[300:301,200:210])
        #mask = mask.eval(session=tf.compat.v1.Session())
        #mask = tf.Session().run(mask)
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        
        #add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.image_ids)
    
    
class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return batch
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
            

#///////////////////////////////// AUGMENTATION \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

# define heavy augmentations
def get_training_augmentation():
    train_transform = [

        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        A.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        A.RandomCrop(height=320, width=320, always_apply=True),

        A.IAAAdditiveGaussianNoise(p=0.2),
        A.IAAPerspective(p=0.5),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.IAASharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.RandomContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(384, 480)
    ]
    return A.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)


if __name__ == '__main__':
    
    args = parser.parse_args()
    
    BACKBONE = args.backbone
    BATCH_SIZE_TRAIN = 4
    BATCH_SIZE_VAL = 4
    #CLASSES = ['car']
    #CLASSES = ['grass']
    #CLASSES = ['hospitals','commercial', 'institutional', 'mosques', 'residential', 'transport&util', 'underconst', 'vegetation', 'others']
    CLASSES = ['built-up', 'underconst', 'vegetation']
    #CLASSES = ['commercial', 'residential']
    '''WEIGHTS = [
        0.15, # Hospitals
        0.1, # Commercial
        0.15, # Institutional
        0.15, # Mosques
        0.05, # Residential
        0.1, # Transport&Util
        0.1, # UnderConst
        0.05, # Vegetation
        0.05, # Others
        0.1 # background
        ]'''
    WEIGHTS = [
        0.02, # Built-up
        0.4, # UnderConst
        0.5, # Vegetation
        0.08 # background
        ]
        
    '''
    WEIGHTS = [
        0.8, # Commercial
        0.1, # Residential
        0.1, # background
        ]
    '''
    LR = 0.0001
    EPOCHS = 20
    
    preprocess_input = sm.get_preprocessing(BACKBONE)
    
    
    # define network parameters
    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
    activation = 'sigmoid' if n_classes == 1 else 'softmax'
    
    #create model
    model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
    
    # define optomizer
    optim = keras.optimizers.Adam(LR)
    
    # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
    dice_loss = sm.losses.DiceLoss(beta=1, class_weights=WEIGHTS, class_indexes=None, per_image=False, smooth=1e-05)
    focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss(WEIGHTS)
    total_loss = dice_loss + (1 * focal_loss)
    
    # actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
    # total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 
    
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
    
    # compile keras model with defined optimozer, loss and metrics
    model.compile(optim, total_loss, metrics)
    
    
    # Lets look at data we have
    
    if (args.mode == 'v'):
        x_train_dir = os.path.join(args.data_dir, 'images')
        y_train_dir = os.path.join(args.data_dir, 'segmentation_v2')
            
        x_valid_dir = os.path.join(args.data_dir, 'test_images')
        y_valid_dir = os.path.join(args.data_dir, 'test_segmentation_v2')
    
        dataset = Dataset(x_train_dir, y_train_dir, classes=CLASSES)
    
        image, mask = dataset[1] # get some sample
        #print(mask)
        
       
        visualize(
            image=image, 
            grass_mask=mask[..., 6].squeeze(),
            pedestrian_mask=mask[..., 7].squeeze()#,
            #background_mask=mask[..., 0].squeeze(),
        )
        
        # Lets look at augmented data we have
        dataset = Dataset(x_train_dir, y_train_dir, classes=CLASSES, augmentation=get_training_augmentation())
        
        image, mask = dataset[12] # get some sample
        
        visualize(
            image=image, 
            grass_mask=mask[..., 6].squeeze(),
            sky_mask=mask[..., 7].squeeze()#,
            #background_mask=mask[..., 2].squeeze(),
        )
        
    if (args.mode=='t' or args.mode=='ct'):
        #//////////////////////// DATASET PATH \\\\\\\\\\\\\\\\\\\\\\\\
        x_train_dir = os.path.join(args.data_dir, 'images')
        y_train_dir = os.path.join(args.data_dir, 'segmentation_v2')
        
        x_valid_dir = os.path.join(args.data_dir, 'test_images')
        y_valid_dir = os.path.join(args.data_dir, 'test_segmentation_v2')

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        else:
            print("Warining Model path {} already exists".format(args.output_dir))
        
        if (args.mode=='ct'):
            model.load_weights(os.path.join(args.output_dir,'best_model.h5'))

        
        # Dataset for train images
        train_dataset = Dataset(
            x_train_dir, 
            y_train_dir, 
            classes=CLASSES, 
            augmentation=get_training_augmentation(),
            preprocessing=get_preprocessing(preprocess_input),
        )
        
        # Dataset for validation images
        valid_dataset = Dataset(
            x_valid_dir, 
            y_valid_dir, 
            classes=CLASSES, 
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(preprocess_input),
        )
        
        train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
        valid_dataloader = Dataloder(valid_dataset, batch_size=BATCH_SIZE_VAL, shuffle=False)
        
        # check shapes for errors
        assert train_dataloader[0][0].shape == (BATCH_SIZE_TRAIN, 320, 320, 3)
        assert train_dataloader[0][1].shape == (BATCH_SIZE_TRAIN, 320, 320, n_classes)
        
        # define callbacks for learning rate scheduling and best checkpoints saving
        callbacks = [
            keras.callbacks.ModelCheckpoint(os.path.join(args.output_dir,'best_model.h5'), save_weights_only=True, save_best_only=True, mode='min'),
            keras.callbacks.ReduceLROnPlateau(),
        ]
        
        # train model
        history = model.fit_generator(
            train_dataloader, 
            steps_per_epoch=len(train_dataloader), 
            #steps_per_epoch=75,
            epochs=EPOCHS, 
            callbacks=callbacks, 
            validation_data=valid_dataloader, 
            validation_steps=len(valid_dataloader),
        )
        
        np.save(os.path.join(args.output_dir,'my_history.npy'),history.history)
        
        #if (args.plot== True):
        # Plot training & validation iou_score values
        plt.figure(figsize=(30, 5))
        plt.subplot(121)
        plt.plot(history.history['iou_score'])
        plt.plot(history.history['val_iou_score'])
        plt.title('Model iou_score')
        plt.ylabel('iou_score')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        
        # Plot training & validation loss values
        plt.subplot(122)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
    
    if (args.mode == 'ph'):
        history=np.load(os.path.join(args.output_dir,'my_history.npy'),allow_pickle='TRUE').item()
        # Plot training & validation iou_score values
        plt.figure(figsize=(30, 5))
        plt.subplot(121)
        plt.plot(history['iou_score'])
        plt.plot(history['val_iou_score'])
        plt.title('Model iou_score')
        plt.ylabel('iou_score')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        
        # Plot training & validation loss values
        plt.subplot(122)
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
    
    if (args.mode=='eb' or args.mode=='es'):
                
        x_test_dir = os.path.join(args.data_dir, 'test_images')
        y_test_dir = os.path.join(args.data_dir, 'test_segmentation_v2')
        
        
        test_dataset = Dataset(
            x_test_dir, 
            y_test_dir, 
            classes=CLASSES, 
            preprocessing=get_preprocessing(preprocess_input),
        )

        test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)
        
        # load best weights
        model.load_weights(os.path.join(args.output_dir,'best_model.h5'))
        
        if (args.mode =='eb'):
            scores = model.evaluate_generator(test_dataloader)
    
            print("Loss: {:.5}".format(scores[0]))
            for metric, value in zip(metrics, scores[1:]):
                print("mean {}: {:.5}".format(metric.__name__, value))
    
        if (args.mode =='es'):
            n = 5
            np.random.seed(230)
            ids = np.random.choice(np.arange(len(test_dataset)), size=n)
            pred_dir = os.path.join(args.data_dir, 'Predictions')
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            
            #for i in ids:
            for i in range(len(test_dataset)):
                image, gt_mask = test_dataset[i]
                image = np.expand_dims(image, axis=0)
                pr_mask = model.predict(image).round().squeeze()
                
                pr_img = np.zeros((pr_mask.shape[0],pr_mask.shape[1],3))
                gt_img = np.zeros((gt_mask.shape[0],pr_mask.shape[1],3))
                '''
                pr_img = pr_img+ MasktoRGB(pr_mask[...,0],Dataset.colors[0])
                gt_img = gt_img+ MasktoRGB(gt_mask[...,0],Dataset.colors[0])
                
                visualize(
                    image=denormalize(image.squeeze()),
                    gt_mask=gt_img.astype(np.uint8),
                    pr_mask=pr_img.astype(np.uint8),
                )
                '''
                for j in range(len(Dataset.CLASSES)):
                    pr_img = pr_img+ MasktoRGB(pr_mask[...,j],Dataset.colors[j])
                    gt_img = gt_img+ MasktoRGB(gt_mask[...,j],Dataset.colors[j])
                
                #img_name = "Image_{}_".format(i)+"_PredMask.png"
                #img_path = os.path.join(pred_dir,img_name)
                #plt.imsave(img_path,pr_img.astype(np.uint8))
                #print(pr_img.astype(int))
                visualize(
                    image=denormalize(image.squeeze()),
                    gt_mask=gt_img.astype(np.uint8),
                    pr_mask=pr_img.astype(np.uint8),
                )
                '''
                visualize(
                    image=denormalize(image.squeeze()),
                    gt_mask=gt_mask[..., 6].squeeze(),
                    pr_mask=pr_mask[..., 6].squeeze(),
                )'''