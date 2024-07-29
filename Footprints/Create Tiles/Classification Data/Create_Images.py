import os
import cv2
import rasterio
import pandas as pd
import numpy as np

tile_size = 512
Down_sample_factor = 1

tiff_path = r"D:\LUMS_RA\Data\Google_Earth_Images_Downloader\Zoom_20\DHA\clipped.tif"
out_fldr_a = r'D:\LUMS_RA\Data\Classification Data\Images\DHA\Building'
out_fldr_b = r'D:\LUMS_RA\Data\Classification Data\Images\DHA\Non_Building'
csv_path = r'D:\LUMS_RA\Data\Classification Data\DHA_Tiles.csv'

dataset = rasterio.open(tiff_path)
x_min_d = dataset.bounds[0]  # left edge
x_max_d = dataset.bounds[2]  # right edge
y_min_d = dataset.bounds[1]  # bottom edge
y_max_d = dataset.bounds[3]  # top edge

tiff_image = np.transpose(dataset.read(), (1, 2, 0))
image_x_pix = tiff_image.shape[1]
image_y_pix = tiff_image.shape[0]

x_delta = (x_max_d - x_min_d) / image_x_pix
y_delta = -1 * (y_max_d - y_min_d) / image_y_pix

x_tiles_num = image_x_pix // tile_size
y_tiles_num = image_y_pix // tile_size


if not os.path.exists(out_fldr_a):
    os.makedirs(out_fldr_a)

if not os.path.exists(out_fldr_b):
    os.makedirs(out_fldr_b)

csv = pd.read_csv(csv_path)
images = csv['filename']
classes = csv['class']
for image, class_ in zip(images, classes):
    fid = int(image)
    x_idx = int(fid / y_tiles_num)
    y_idx = fid % y_tiles_num

    x1 = tile_size * x_idx
    x2 = x1 + tile_size
    y1 = tile_size * y_idx
    y2 = y1 + tile_size

    img_crop = tiff_image[y1:y2, x1:x2]
    final_size = int(tile_size * Down_sample_factor)
    img = cv2.resize(img_crop,(final_size, final_size))

    if class_ == 'Building':
        cv2.imwrite(os.path.join(out_fldr_a, '{}.png'.format(fid)), img)
    else:
        cv2.imwrite(os.path.join(out_fldr_b, '{}.png'.format(fid)), img)