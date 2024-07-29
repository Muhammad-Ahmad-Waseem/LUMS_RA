import os
import cv2
import rasterio
import pandas as pd
import subprocess
import numpy as np

tile_size = 512
tiff_path = r"D:\LUMS_RA\Data\Google_Earth_Images_Downloader\Zoom_20\Phae1_2\clipped.tif"

images_folder = 'images'
geo_images_folder = 'georeferenced_images'

base_path = r'D:\LUMS_RA\Data\Seg_data\DHA_Ph12\try'
partitions = [folder for folder in os.listdir(base_path) if '.' not in folder]

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

for partition in partitions:
    csv_path = os.path.join(base_path, partition, "{}_tiles.csv".format(partition))
    out_fldr = os.path.join(base_path, partition, images_folder)
    out_fldr2 = os.path.join(base_path, partition, geo_images_folder)

    if not os.path.exists(out_fldr):
        os.makedirs(out_fldr)

    if not os.path.exists(out_fldr2):
        os.makedirs(out_fldr2)

    csv = pd.read_csv(csv_path)
    images = csv['filename']
    for image in images:
        fid = int(image)
        x_idx = int(fid / y_tiles_num)
        y_idx = fid % y_tiles_num

        x1 = tile_size * x_idx + tile_size//2
        x2 = x1 + tile_size
        y1 = tile_size * y_idx + tile_size//2
        y2 = y1 + tile_size

        img_crop = tiff_image[y1:y2, x1:x2]
        # print("[{}:{}, {}:{}]".format(y1,y2,x1,x2))
        # print(img_crop)
        cv2.imwrite(os.path.join(out_fldr, '{}.png'.format(fid+390)), img_crop)

        x_min = float(x_min_d + x1 * x_delta)
        x_max = float(x_min_d + x2 * x_delta)
        y_max = float(y_max_d + y1 * y_delta)
        y_min = float(y_max_d + y2 * y_delta)

        gcp1 = "{} {} {} {}".format(0, 0, x_min, y_max)
        gcp2 = "{} {} {} {}".format(512, 0, x_max, y_max)
        gcp3 = "{} {} {} {}".format(512, 512, x_max, y_min)
        gcp4 = "{} {} {} {}".format(0, 512, x_min, y_min)

        command1 = "gdal_translate -of GTiff -gcp {} -gcp {} -gcp {} -gcp {} {} {}".format(
            gcp1, gcp2, gcp3, gcp4, os.path.join(out_fldr, '{}.png'.format(fid+390)), os.path.join(base_path,
                                                                                                '{}.png'.format(fid+390))
        )

        command2 = "gdalwarp -r near -order 1 -co COMPRESS=NONE -t_srs EPSG:3857 {} {}".format(
            os.path.join(base_path, '{}.png'.format(fid+390)), os.path.join(out_fldr2, '{}.tif'.format(fid+390))
        )

        subprocess.call(command1, shell=True)
        subprocess.call(command2, shell=True)
        subprocess.check_output("del {}".format(os.path.join(base_path, '{}.png'.format(fid+390))), shell=True)

    # print(len(images))
    # print(images[0])