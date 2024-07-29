import os
import argparse
import matplotlib.pyplot as plt

import numpy as np
import segmentation_models_pytorch as smp
# from segmentation_models_pytorch import utils as smp_utils
import torch
from tqdm import tqdm as tqdm
import sys

from Dataset import Dataset
from torch.utils.data import DataLoader
from Augmentations import *
# import utils

#//////////////////////// ARGUMENT PARSER \\\\\\\\\\\\\\\\\\\\\\\\
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default="/mnt/Ahmad/Building_Footprints_Extraction/Benchmarks/rgb-footprint-extract/data/SpaceNet/Vegas/test", help='Path to base data directory')
parser.add_argument('--output_dir', default="/mnt//Ahmad//Models//Segmentation//trained_model//built_vs_unbuilt_FL_SN", help="Path to save model")
parser.add_argument('--backbone', default='efficientnetb3')
parser.add_argument('--b_train', default=8, help='Batch size for training')
parser.add_argument('--b_valid', default=1, help='Batch size for validation')
parser.add_argument('--lr', default=0.0001, help='learning rate')
parser.add_argument('--epochs', default=100, help='Total training epochs')
parser.add_argument('--th', default=0.5, help='Threshold for computation of tp,fp,fn,tn')
parser.add_argument('--encoder', default='resnet50', help='Encoder to use')
parser.add_argument('--encoder_weights', default='imagenet', help='Pre-Trained weights for encoder')
parser.add_argument('--ct_train', action='store_true', default=True, help='Continue training')
parser.add_argument('--overwrite', action='store_true', default=False, help='Overwrite original model')
parser.add_argument('--db', default="SpaceNet", help='Overwrite original model')



'''
To define Loss function, types required are as follows:

Ground Truth : torch.tensor of shape(N,1,H,W) for binary mode, (N,H,W) for multiclass mode and (N,C,H,W) 
               for multilabel mode.
Prediction : (N,1,H,W) for binary mode, (N,C,H,W) for multiclass mode and multilabel mode.

See smp.losses for more details. For details of mode, see documentation of  smp library.
'''

#////////////////////////Pre Processing\\\\\\\\\\\\\\\\\\\\\\\\
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
    CLASSES = ['built-up']
    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)
    mode = 'binary' if len(CLASSES) == 1 else 'multiclass'
    mode_choices = ['binary', 'multiclass', 'multilabel'] #See documentation for more details.
    ACTIVATION = 'sigmoid' if n_classes == 1 else 'softmax2D'  # could be None for logits or 'softmax2d' for multiclass segmentation
    model = smp.DeepLabV3Plus(
        encoder_name=args.encoder,
        encoder_weights=args.encoder_weights,
        classes=n_classes,
        activation=ACTIVATION
    )
    print(model)

    # Pre-processing function to pre-process whole data for encoder
    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder, args.encoder_weights)

    # define optimizer
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=args.lr),
    ])

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=1, T_mult=2, eta_min=0.000005,
    )

    focal_loss = smp.losses.FocalLoss(mode)
    # jacc_loss = smp_utils.losses.JaccardLoss()

    loss = focal_loss  # (_lambda * jacc_loss)

    # Define metrics to compute
    metrics = {"IOU Score": smp.metrics.iou_score,
               "F1 Score": smp.metrics.f1_score,
               "F_beta score": smp.metrics.fbeta_score,
               "Accuracy": smp.metrics.accuracy,
               "Recall": smp.metrics.recall}

    # Set device: `cuda` or `cpu`
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    if args.ct_train:
        assert os.path.exists(os.path.join(args.output_dir, 'best_model.h5')), "Model not found at path: {}".format(
            args.output_dir)
        model = torch.load(os.path.join(args.output_dir, 'best_model.h5'), map_location=DEVICE)
        print('Loaded pre-trained DeepLabV3+ model!')
    else:
        if os.path.exists(os.path.join(args.output_dir, 'best_model.h5')):
            assert args.overwrite, "Path already exists, cannot over write. Please set the overwrite flag to high,\
            if you want to overwrite"
            print("Warining Model path {} already exists, overwriting it".format(args.output_dir))
        else:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

    model.to(DEVICE)

    if args.db == 'TeamUrbanTech':
        # Dataset for train images
        train_dataset = Dataset(
            args.data_dir,
            classes=CLASSES,
            augmentation=get_training_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn),
            train=True
        )
        # Dataset for validation images
        valid_dataset = Dataset(
            args.data_dir,
            classes=CLASSES,
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn),
            train=False
        )
    elif args.db == 'SpaceNet':
        # Dataset for train images
        train_dataset = Dataset(
            args.data_dir,
            classes=CLASSES,
            augmentation=get_training_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn),
            db=args.db,
            train=True
        )
        # Dataset for validation images
        valid_dataset = Dataset(
            args.data_dir,
            classes=CLASSES,
            augmentation=get_validation_augmentation(),
            db=args.db,
            preprocessing=get_preprocessing(preprocessing_fn),
            train=False
        )
    else:
        assert False, "The provided database is not supported"

    threshold = 0.5  # Threshold for computation of tp,fp,fn,tn
    train_loader = DataLoader(train_dataset, batch_size=args.b_train, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=args.b_valid, shuffle=False, num_workers=0)

    best_iou_score = 0.0
    # break_value = 3
    # counter = 0
    if args.ct_train:
        f = open(os.path.join(args.output_dir, 'train_logs.txt'), "r")
        lines = f.readlines()
        f.close()
        if len(lines) > 1:
            start = int(lines[-1].split(' ')[0]) + 1
        else:
            start = 0

        print("Starting from Epoch {}".format(start))
    else:
        with open(os.path.join(args.output_dir, 'train_logs.txt'), "w") as f:
            f.write("Epoch #: Logs" + "\n")

        with open(os.path.join(args.output_dir, 'valid_logs.txt'), "w") as f:
            f.write("Epoch #: Logs" + "\n")

        start = 0

    print("Total Epochs: {}".format(args.epochs))
    for i in range(start, args.epochs):
        # Perform training & validation
        print('\nEpoch: {}'.format(i))
        model.train()
        train_logs = {}
        loss_meter = np.array([])
        metrics_meters = {name: np.array([]) for name, func in metrics.items()}
        verbose = True

        with tqdm(train_loader) as iterator:
            for x, y, _ in iterator:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                y_pred = model.forward(x)
                loss_Val = loss(y_pred, y)
                loss_Val.backward()
                optimizer.step()
                # update loss logs
                loss_value = loss_Val.cpu().detach().numpy()
                loss_meter = np.append(loss_meter, loss_value)
                loss_logs = {"LOSS": np.mean(loss_meter)}
                train_logs.update(loss_logs)

                if mode == mode_choices[2]:
                    tp, fp, fn, tn = smp.metrics.get_stats(torch.argmax(y_pred, dim=1), y, mode=mode,
                                                           threshold=threshold)
                elif mode == mode_choices[1]:
                    tp, fp, fn, tn = smp.metrics.get_stats(torch.argmax(y_pred, dim=1), y, mode=mode,
                                                           num_classes=n_classes)
                elif mode == mode_choices[0]:
                    tp, fp, fn, tn = smp.metrics.get_stats(torch.squeeze(y_pred, dim=1), y, mode=mode,
                                                           threshold=threshold)

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
        metrics_meters = {name: np.array([]) for name, func in metrics.items()}
        verbose = True

        with tqdm(valid_loader, desc="Valid", file=sys.stdout, disable=not verbose) as iterator:
            for x, y, _ in iterator:
                x, y = x.to(DEVICE), y.to(DEVICE)
                with torch.no_grad():
                    y_pred = model.forward(x)
                    loss_Val = loss(y_pred, y)

                # update loss logs
                loss_value = loss_Val.cpu().detach().numpy()
                loss_meter = np.append(loss_meter, loss_value)
                loss_logs = {"LOSS": np.mean(loss_meter)}
                valid_logs.update(loss_logs)

                if mode == mode_choices[2]:
                    tp, fp, fn, tn = smp.metrics.get_stats(torch.argmax(y_pred, dim=1), y, mode=mode,
                                                           threshold=threshold)
                elif mode == mode_choices[1]:
                    tp, fp, fn, tn = smp.metrics.get_stats(torch.argmax(y_pred, dim=1), y, mode=mode,
                                                           num_classes=n_classes)
                elif mode == mode_choices[0]:
                    tp, fp, fn, tn = smp.metrics.get_stats(torch.squeeze(y_pred, dim=1), y, mode=mode,
                                                           threshold=threshold)

                # update metrics logs
                for name, metric_fn in metrics.items():
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
