#----------------------------------------------------------------------------------------------------
'''
This is the code for 3 class segmentation: object mask (building), boundary, background.
Goal is to be able to detect buildings' footprints using semantic segmenetation. The library used is
segmentation models pytorch (smp, documentation: https://smp.readthedocs.io/en/latest) and code is using pytorch for implementation.
'''
#----------------------------------------------------------------------------------------------------

#import from installed libraries
import os
import argparse
import matplotlib.pyplot as plt
#from PIL import Image
import albumentations
import numpy as np
import segmentation_models_pytorch as smp
import torch
from tqdm import tqdm as tqdm
import sys
from torch.utils.data import DataLoader

#import from local files
from Dataset import Dataset
import utils

#//////////////////////// ARGUMENT PARSER \\\\\\\\\\\\\\\\\\\\\\\\
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='D://LUMS_RA//Seg_data')
parser.add_argument('--output_dir', default='D://LUMS_RA//Models//Segmentation//trained_model//DeepLabV3+')
parser.add_argument('--backbone', default='efficientnetb3')
parser.add_argument('-m',   '--mode', choices=['t', 'es', 'ct', 'v', 'ea'], default='t', required='true')

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

# define heavy augmentations
def get_training_augmentation():
    train_transform = [albumentations.HorizontalFlip(p=0.5),
        albumentations.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        albumentations.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albumentations.RandomCrop(height=320, width=320, always_apply=True),
        albumentations.IAAAdditiveGaussianNoise(p=0.2),
        #albumentations.IAAPerspective(p=0.5),
        albumentations.OneOf(
            [
                albumentations.CLAHE(p=1),
                albumentations.RandomBrightness(p=1),
                albumentations.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        albumentations.OneOf(
            [
                albumentations.IAASharpen(p=1),
                albumentations.Blur(blur_limit=3, p=1),
                albumentations.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        albumentations.OneOf(
            [
                albumentations.RandomContrast(p=1),
                albumentations.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        albumentations.Lambda(mask=round_clip_0_1)
    ]
    
    return albumentations.Compose(train_transform)

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albumentations.PadIfNeeded(384, 480)
    ]
    return albumentations.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albumentations.Lambda(image=preprocessing_fn),
    ]
    return albumentations.Compose(_transform)
    
if __name__ == '__main__':
    args = parser.parse_args()
    # define different parameters to be used parameters
    n_classes = 3
    BACKBONE = args.backbone
    BATCH_SIZE_TRAIN = 32
    BATCH_SIZE_VAL = 16
    LR = 0.00004
    EPOCHS = 40
    mode = 'multiclass'
    mode_choices = ['multiclass', 'multilabel'] #See documentation for more details.
    threshold = 0.5 #Threshold for computation of tp,fp,fn,tn

    assert (mode in mode_choices), "Mode can be only from one of these: {}".format(mode_choices)
    #Define Network/ Models ans it's parameters. See documentation for more details.
    ENCODER = 'resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=n_classes, 
        activation=ACTIVATION
        )
    #print(model)

    #Pre-processing function to pre-process whole data for encoder
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    
    # define optimizer
    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=LR),
    ])
    
    '''
    Define Loss function. See smp.losses for more details:
    Types required are as follows:
    Ground Truth : torch.tensor of shape(N,1,H,W) for binary mode, (N,H,W) for multiclass mode and (N,C,H,W) 
                   for multilabel mode.
    Prediction : (N,1,H,W) for binary mode, (N,C,H,W) for multiclass mode and multilabel mode.
    
    To see details of mode, see documentation of smp library.
    '''
    dice_loss = smp.losses.DiceLoss(mode='multiclass')
    loss = dice_loss

    #Define metrics to compute
    metrics = {"IOU Score": smp.metrics.iou_score,
               "F1 Score": smp.metrics.f1_score,
               "F_beta score": smp.metrics.fbeta_score,
               "Accuracy": smp.metrics.accuracy,
               "Recall": smp.metrics.recall}

    # Set device: `cuda` or `cpu`
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    # Lets look at data we have
    if (args.mode == 'v'):
        imgs_dir = os.path.join(args.data_dir, 'images')
        segm_dir = os.path.join(args.data_dir, 'segmentations')
        contr_dir = os.path.join(args.data_dir, 'contours')
            
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dataset = Dataset(
            imgs_dir, 
            segm_dir,
            contr_dir,
            preprocessing=get_preprocessing(preprocessing_fn),
        )

        image, mask = dataset[1]
        print(mask.dtype)
        #print(image.shape)
        #print(mask.shape)
        plt.imshow(mask[0]+2*mask[1]+3*mask[2])
        plt.show()

    # Train the model from start or from initial point
    if (args.mode=='t' or args.mode=='ct'):
        #//////////////////////// DATASET PATH \\\\\\\\\\\\\\\\\\\\\\\\
        images_train_dir = os.path.join(args.data_dir, 'images')
        buildings_train_dir = os.path.join(args.data_dir, 'segmentations')
        boundaries_train_dir = os.path.join(args.data_dir, 'contours')

        images_test_dir = os.path.join(args.data_dir, 'test_images')
        buildings_test_dir = os.path.join(args.data_dir, 'test_segmentations')
        boundaries_test_dir = os.path.join(args.data_dir, 'test_contours')
        
        if args.mode == 't':
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            else:
                print("Warining Model path {} already exists".format(args.output_dir))
        
        if args.mode == 'ct':
            assert os.path.exists(args.output_dir), "Model not found at path: {}".format(args.output_dir)
            model = torch.load(os.path.join(args.output_dir,'best_model.h5'), map_location=DEVICE)
            print('Loaded pre-trained DeepLabV3+ model!')

        # Dataset for train images
        train_dataset = Dataset(
            images_train_dir,
            buildings_train_dir,
            boundaries_train_dir,
            augmentation=get_training_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn),
        )
        
        # Dataset for validation images
        valid_dataset = Dataset(
            images_test_dir,
            buildings_test_dir,
            boundaries_test_dir,
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn),
        )
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=8)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE_VAL, shuffle=False, num_workers=4)

        best_iou_score = 0.0
        print("Total Epochs: {}".format(EPOCHS))
        for i in range(0, EPOCHS):
            # Perform training & validation
            print('\nEpoch: {}'.format(i))
            model.train()
            train_logs = {}
            loss_meter = np.array([])
            metrics_meters = {name : np.array([]) for name, func in metrics.items()}
            verbose = True

            with open(os.path.join(args.output_dir, 'train_logs.txt'), "w") as f:
                f.write("Epoch #: Logs" + "\n")

            with open(os.path.join(args.output_dir, 'valid_logs.txt'), "w") as f:
                f.write("Epoch #: Logs" + "\n")

            with tqdm(train_loader, desc="Train", file=sys.stdout, disable=not verbose) as iterator:
                for x, y in iterator:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    optimizer.zero_grad()
                    y_pred = model.forward(x)
                    loss_Val = loss(y_pred, y)
                    loss_Val.backward()
                    optimizer.step()
                    # update loss logs
                    loss_value = loss_Val.cpu().detach().numpy()
                    loss_meter = np.append(loss_meter, loss_value)
                    loss_logs = {"DICE_LOSS": np.mean(loss_meter)}
                    train_logs.update(loss_logs)

                    if mode == mode_choices[1]:
                        tp, fp, fn, tn = smp.metrics.get_stats(torch.argmax(y_pred,dim=1), y, mode=mode,
                                                               threshold=threshold)
                    else:
                        tp, fp, fn, tn = smp.metrics.get_stats(torch.argmax(y_pred, dim=1), y, mode=mode,
                                                               num_classes=n_classes)

                    # update metrics logs
                    for name, metric_fn in metrics.items():
                        metric_value = metric_fn(tp, fp, fn, tn).cpu().detach().numpy()
                        metrics_meter = metrics_meters[name]
                        metrics_meters[name] = np.append(metrics_meter, metric_value)

                    metrics_logs = {name: np.mean(values) for name, values in metrics_meters.items()}
                    train_logs.update(metrics_logs)

                    if verbose:
                        str_logs = ['{} - {:.4}'.format(k, v) for k, v in train_logs.items()]
                        s = ', '.join(str_logs)
                        iterator.set_postfix_str(s)

            torch.save(model, os.path.join(args.output_dir, 'best_model.h5'))

            with open(os.path.join(args.output_dir, 'train_logs.txt'), 'a') as f:
                f.write("{} : {}".format(i, str(train_logs)) + "\n")

            model.eval()
            valid_logs = {}
            loss_meter = np.array([])
            metrics_meters = {name : np.array([]) for name, func in metrics.items()}
            verbose = True

            with tqdm(valid_loader, desc="Valid", file=sys.stdout, disable=not verbose) as iterator:
                for x, y in iterator:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    with torch.no_grad():
                        y_pred = model.forward(x)
                        loss_Val = loss(y_pred, y)

                    # update loss logs
                    loss_value = loss_Val.cpu().detach().numpy()
                    loss_meter = np.append(loss_meter, loss_value)
                    loss_logs = {"DICE_LOSS": np.mean(loss_meter)}
                    valid_logs.update(loss_logs)

                    if mode == mode_choices[1]:
                        tp, fp, fn, tn = smp.metrics.get_stats(torch.argmax(y_pred, dim=1), y, mode=mode,
                                                               threshold=threshold)
                    else:
                        tp, fp, fn, tn = smp.metrics.get_stats(torch.argmax(y_pred, dim=1), y, mode=mode,
                                                               num_classes=n_classes)


                    # update metrics logs
                    for name,metric_fn in metrics.items():
                        metric_value = metric_fn(tp, fp, fn, tn).cpu().detach().numpy()
                        metrics_meter = metrics_meters[name]
                        metrics_meters[name] = np.append(metrics_meter, metric_value)

                    metrics_logs = {name: np.mean(vals) for name, vals in metrics_meters.items()}
                    valid_logs.update(metrics_logs)

                    if verbose:
                        str_logs = ['{} - {:.4}'.format(k, v) for k, v in valid_logs.items()]
                        s = ', '.join(str_logs)
                        iterator.set_postfix_str(s)

            with open(os.path.join(args.output_dir, 'valid_logs.txt'), 'a') as f:
                f.write("{} : {}".format(i, str(valid_logs)) + "\n")


    #Predict against all images in a folder and save results as images
    if(args.mode=='ea'):
        imgs_dir = os.path.join(args.data_dir, 'images')
        pred_dir = os.path.join(args.data_dir, 'predictions')
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load best weights
        model = torch.load(os.path.join(args.output_dir,'best_model.h5'), map_location=DEVICE)
        dataset = Dataset(
            imgs_dir, 
            imgs_dir, 
            classes=CLASSES, 
            preprocessing=get_preprocessing(preprocessing_fn),
        )
        i = 0
        with tqdm(dataset, file=sys.stdout) as iterator:
            for it in iterator:
                image, _,f_name = dataset[i]
                x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
                image = image.squeeze()
                image = np.transpose(image,(1,2,0))
                pred_mask = model(x_tensor)
                pr_mask = pred_mask.round().squeeze()
                pr_mask = pr_mask.detach().squeeze().cpu().numpy()
                
                if mode == smp.losses.constants.BINARY_MODE:
                    pr_mask = np.expand_dims(pr_mask, 0)
                
                #print(pr_mask.shape)
                pr_mask = np.transpose(pr_mask,(1,2,0))
                pr_img = np.zeros((pr_mask.shape[0],pr_mask.shape[1],3))
                for j in range(pr_mask.shape[2]):
                    pr_img = pr_img+ utils.MasktoRGB(pr_mask[...,j],Dataset.colors[j])
                    
                #f_name = "{}.png".format(i+1)
                save_path = os.path.join(pred_dir,f_name)
                plt.imsave(save_path,pr_img.astype(np.uint8))
                i = i+1
                #print("Output for Image: {} has been saved!".format(f_name))

    #Predict and show result for only a single image (doesn't save)
    if (args.mode=='es'):
                
        images_dir = os.path.join(args.data_dir, 'test_images')
        masks_dir = os.path.join(args.data_dir, 'test_segmentations')
        contour_dir = os.path.join(args.data_dir, 'test_contours')
        
        # Set device: `cuda` or `cpu`   
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_dataset = Dataset(
            images_dir,
            masks_dir,
            contour_dir,
            preprocessing=get_preprocessing(preprocessing_fn),
        )
        
        # load best weights
        best_model = torch.load(os.path.join(args.output_dir,'best_model.h5'), map_location=DEVICE)

        image, gt_mask = test_dataset[0]
        #gt_mask = np.transpose(gt_mask,(1,2,0))

        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        #image = image.squeeze()
        image = np.transpose(image,(1,2,0))

        pred_mask = best_model(x_tensor)
        pr_mask = pred_mask.squeeze()
        pr_mask = pr_mask.detach().squeeze().cpu().numpy()
        pr_mask = np.argmax(pr_mask, axis=0)

        #print(pr_mask.shape)
        #pr_mask = np.transpose(pr_mask,(1,2,0))
        '''
        print(pr_mask.shape)
        print(gt_mask.shape)
        print(image.shape)'''
        gt_img = 3*gt_mask[0]+2*gt_mask[1]+gt_mask[2]
        pr_img = 3*(pr_mask == 0) + 2*(pr_mask == 1) + (pr_mask == 2)
        print(pr_img.shape)

        #print(name)
        utils.visualize(
            image=utils.denormalize(image.squeeze()),
            gt_mask=gt_img.astype(np.uint8),
            pr_mask=pr_img.astype(np.uint8),
        )
