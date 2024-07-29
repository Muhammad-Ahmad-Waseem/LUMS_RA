import rasterio
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import subprocess
import argparse
import numpy as np
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--refernce_tiff', default='test.tiff')
parser.add_argument('--input_images', default='./*.png')
parser.add_argument('--output_dir', default='./Outputs')

if __name__ == '__main__':

    args = parser.parse_args()
    dataset = rasterio.open(args.refernce_tiff)
    x_min_coord = dataset.bounds[0]  # left edge
    x_max_coord = dataset.bounds[2]  # right edge
    y_min_coord = dataset.bounds[1]  # bottom edge
    y_max_coord = dataset.bounds[3]  # top edge

    imgs = [file for file in glob.glob(args.input_images) if file.endswith('.png')]
    pos = len(args.input_images.split('*')[0])
    for img in imgs:
        height, width, _ = np.array(Image.open(img)).shape

        sub_dirs = os.path.split(img[pos:])[0]
        out_path = os.path.join(args.output_dir, sub_dirs)
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        file_name, ext = os.path.split(img)[-1].split('.')
        temp_file = os.path.join(out_path, file_name+"_temp."+ext)
        outputfile = os.path.join(out_path, file_name+".tif")

        gcp1 = "{} {} {} {}".format(0, 0, x_min_coord, y_max_coord)
        gcp2 = "{} {} {} {}".format(width, 0, x_max_coord, y_max_coord)
        gcp3 = "{} {} {} {}".format(width, height, x_max_coord, y_min_coord)
        gcp4 = "{} {} {} {}".format(0, height, x_min_coord, y_min_coord)

        command1 = "gdal_translate -of GTiff -gcp {} -gcp {} -gcp {} -gcp {} {} {}".format(gcp1, gcp2, gcp3, gcp4,
                                                                                           img, temp_file)
        command2 = "gdalwarp -r near -order 1 -co COMPRESS=NONE  -t_srs EPSG:4326 {} {}".format(temp_file, outputfile)

        subprocess.call(command1, shell=True)
        subprocess.call(command2, shell=True)

