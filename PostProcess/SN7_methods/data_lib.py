import os
import sys
import multiprocessing
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import skimage
import gdal
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

import solaris as sol
from solaris.raster.image import create_multiband_geotiff
from solaris.utils.core import _check_gdf_load

module_path = os.path.abspath(os.path.join('./src/'))
if module_path not in sys.path:
    sys.path.append(module_path)
from solaris.preproc.image import LoadImage, SaveImage, Resize
from sn7_baseline_prep_funcs import map_wrapper, make_geojsons_and_masks


# ###### common configs for divide images ######

# pre resize
pre_height = None  # 3072
pre_width = None  # 3072
# final output size
target_height = 256
target_width = 256
# stride
height_stride = 256
width_stride = 256
#Target Size
target_size = (256,256)
#padding pixels
padding_pixels = 32
padding_value = 0


# ###########################


def get_color_map_list(num_classes):
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j, lab = 0, i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    return color_map


def compose_arr(divide_img_ls, src_img_dir, compose_img_dir, ext=".npy"):
    """
    Core function of putting results into one.
    """
    im_list = sorted(divide_img_ls)
    last_file = os.path.split(im_list[-1])[-1]
    file_name = '_'.join(last_file.split('.')[0].split('_')[:-2])
    yy, xx = last_file.split('.')[0].split('_')[-2:]
    rows = int(yy) // height_stride + 1
    cols = int(xx) // width_stride + 1
    #print(rows)
    #print(cols)
    sizex,sizey = target_size
    
    image = np.zeros((rows * sizex, cols * sizey), dtype=np.float32) * 255
    #print(image.shape)
    for y in range(rows):
        for x in range(cols):
            patch = np.load(im_list[cols * y + x])
            #print(patch.shape)
            assert patch.shape == target_size
            
            #print("{},{}".format(y,x))
            image[y * sizey: (y + 1) * sizey, x * sizex: (x + 1) * sizex] = patch

    img = np.array(Image.open(os.path.join(src_img_dir,file_name+".tif")))
    src_im_height = img.shape[0]
    src_im_width = img.shape[1]
    
    if ext == ".png":
        plt.imsave(os.path.join(compose_img_dir, file_name + '.png'),image.round().astype(np.uint8))
    elif ext == ".npy":
        image = image[:src_im_height, :src_im_width]
        np.save(os.path.join(compose_img_dir, file_name + ".npy"), image)
        print("Image saved with size {}".format(image.shape))


def divide_img(img_file, save_dir, inter_type=cv2.INTER_LINEAR, ext=".png"):
    """
    Core function of dividing images.
    img_file contains img location
    save_dir contains save location
    """

    img = np.array(Image.open(img_file))
    img = cv2.copyMakeBorder(img, padding_pixels, padding_pixels, padding_pixels, padding_pixels,
                                              cv2.BORDER_CONSTANT, value=padding_value)
    src_im_height = img.shape[0]
    src_im_width = img.shape[1]
    
    x1, y1, idx = 0, 0, 0
    while y1 < src_im_height:
        y2 = y1 + target_height + 2*padding_pixels
        while x1 < src_im_width:
            x2 = x1 + target_width + 2*padding_pixels
            img_crop = img[y1: y2, x1: x2]
            pad_bottom = y2 - src_im_height if y2 > src_im_height else 0
            pad_right = x2 - src_im_width if x2 > src_im_width else 0

            if pad_bottom > 0 or pad_right > 0:
                img_crop = cv2.copyMakeBorder(img_crop, 0, pad_bottom, 0, pad_right,
                                              cv2.BORDER_CONSTANT, value=padding_value)
            save_file = os.path.join(save_dir.split(".")[0] + "_%05d_%05d" % (y1, x1) + ext)
            Image.fromarray(img_crop).save(save_file)
            x1 += width_stride
            idx += 1
        x1 = 0
        y1 += height_stride


def divide(root):
    """
    Considering the training speed, we divide the image into small images.
    """
    images_dir = os.path.join(root, "Images")
    divide_dir = os.path.join(root, "Images_divide")
    
    if not os.path.exists(divide_dir):
        os.makedirs(divide_dir)
        
    paths = [os.path.join(images_dir, x) for x in os.listdir(images_dir)]
    img_paths = [path for path in paths if not os.path.isdir(path)]
    
    print("Found {} Images. Starting Divide...".format(len(img_paths)))
    counter = 0
    for img in img_paths:
        divide_img(img, os.path.join(divide_dir, os.path.split(img)[-1]))
        counter = counter+1
        print("Done {} of {}".format(counter,len(img_paths)))


def compose(root,ext=".npy"):
    """
    Because the images are cut into small parts, the output results are also small parts.
    We need to put the output results into a large one.
    """
    dst = os.path.join(root, "vis_compose")
    src = os.path.join(root, "Images")
    vis = os.path.join(root, "vis")
    
    if not os.path.exists(dst):
        os.makedirs(dst)
    dic = {}
    
    img_files = [os.path.join(vis, x) for x in os.listdir(vis)]
    for img_file in img_files:
        key = '_'.join(os.path.split(img_file)[-1].split('_')[:-2])
        if key not in dic:
            dic[key] = [img_file]
        else:
            dic[key].append(img_file)

    for k, v in dic.items():
        print(k)
        #return
        compose_arr(v,src, dst,ext)



def create_test_list(root):
    """
    Create test list.
    """
    fw = open("test_list.txt", 'w')
    _dir = os.path.join(root, "Images_divide")
    for file in os.listdir(_dir):
        if os.path.isdir(os.path.join(_dir, file)):
            continue
        img_path = os.path.join(_dir, file)
        fw.write(img_path + "\n")
    fw.close()
