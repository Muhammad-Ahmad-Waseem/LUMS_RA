import argparse
import random
import os
import numpy as np

from PIL import Image
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--images_dir', default="/home/gcf/Desktop/Ahmad/images", help= "Directory with the images for dataset")
parser.add_argument('--masks_dir', default="/home/gcf/Desktop/Ahmad/masks", help= "Directory with the masks for dataset")
parser.add_argument('--contour_dir', default="/home/gcf/Desktop/Ahmad/contours", help= "Directory with the contours for dataset")
parser.add_argument('--output_dir', default="/home/gcf/Desktop/Ahmad/Tiles", help="Where to write the new data")
parser.add_argument('--seed', default=230, help="The seed value for randomizer")
parser.add_argument('--split', default=0.2, help="The split for validation set")



def save(filename, output_dir):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = Image.open(filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    #image = image.resize((size, size), Image.BILINEAR)
    image.save(os.path.join(output_dir, filename.split('/')[-1]))


if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.images_dir), "Couldn't find the dataset at {}".format(args.images_dir)
    assert os.path.isdir(args.masks_dir), "Couldn't find the dataset at {}".format(args.masks_dir)
    

    images_name = os.listdir(args.images_dir)
    images_name.sort()
    masks_name = os.listdir(args.masks_dir)
    masks_name.sort()
    contour_name = os.listdir(args.contour_dir)
    contour_name.sort()
    
    random.seed(args.seed)
    c = list(zip(images_name, masks_name, contour_name))
    random.shuffle(c)
    
    images_name, masks_name, contour_name = zip(*c)
    
    images_path = [os.path.join(args.images_dir, f) for f in images_name if f.endswith('.png')]
    masks_path = [os.path.join(args.masks_dir, f) for f in masks_name if f.endswith('.png')]
    contour_path = [os.path.join(args.contour_dir, f) for f in contour_name if f.endswith('.png')]
    
    #images_path.sort()
    #masks_path.sort()
    
    train_index = int((1 - args.split) * len(images_path))
    #val_index   = int(0.9 * len(class_images_path))
    train_filenames_img = images_path[:train_index]
    train_filenames_msk = masks_path[:train_index]
    train_filenames_ctr = contour_path[:train_index]
    #val_filenames   = class_images_path[train_index:val_index]
    test_filenames_img = images_path[train_index:]
    test_filenames_msk = masks_path[train_index:]
    test_filenames_ctr = contour_path[train_index:]
    
    filenames = {'images': train_filenames_img,'masks': train_filenames_msk, 'contours': train_filenames_ctr, 'test_images': test_filenames_img,'test_masks': test_filenames_msk, 'test_contours': test_filenames_ctr}
    
    #filenames = {'segmentations2': train_filenames_msk,'test_segmentations2': test_filenames_msk}
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))
        
    for folder in ['test_masks', 'test_contours']:
        print(folder)
        output_dir_split = os.path.join(args.output_dir, '{}'.format(folder))
        if not os.path.exists(output_dir_split):
            #print('here')
            os.makedirs(output_dir_split)
            
        print("Processing data, saving preprocessed data to {}".format(folder))
        for filename in tqdm(filenames[folder]):
            save(filename, output_dir_split)
            #print(filename)

    print("Done building dataset")
