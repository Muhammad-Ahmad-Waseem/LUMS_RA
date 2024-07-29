"""
This is a custom code written for converting large tiff images into smaller tiles of fixed pixel size that can be used
for training the images. Provided the paths of tiff image, the tile size and output directories. It creates tiff tiles
(geo-referenced) from large image and also creates a csv file containing polygon geometry (geo-referenced) against each
tile.

Required Inputs: Tiff Image, CRS of tiff image, tile size you need (in pixels), and the directory where you want results
to be stored.
Written By: Muhammad Ahmad Waseem
"""


# import all the required libraries
import rasterio
import os
import math
import cv2
import numpy as  np
from rasterio.windows import Window
from rasterio.transform import from_origin
import pyproj

# This is the crs of iur rasters. Change it to something else if you want
crs = pyproj.CRS.from_epsg(3857)

# Get all the tiff files in the provided folder
tiff_files_path = r'D:\Ahmad\Satellite Data'
tiff_files = [file for file in os.listdir(tiff_files_path) if file.endswith('tif')]

# Select the tile size for each small (in pixels)
tile_size = 512
expected_tile_size = (3, tile_size, tile_size)

# The output directory (main). Images will be placed in a sub-folder for each tiff file found.
tiles_out_dir = r'D:\Ahmad\Tiles'

# Iterate through all the founded tiff files
for tiff_file in tiff_files:
    # Read the name of tiff folder and create a folder
    folder_name = tiff_file.split('.')[0]
    folder_path = os.path.join(tiles_out_dir, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # create a sub-folder to save geo-referenced images
    tiles_tif_out_dir = os.path.join(folder_path, 'georeferenced_images')
    if not os.path.exists(tiles_tif_out_dir):
        os.makedirs(tiles_tif_out_dir)

    # csv will be saved in main folder for the tiff file
    tiles_csv_out_dir = folder_path
        
    input_tiff_path = os.path.join(tiff_files_path, tiff_file)
    input_tiff_raster = rasterio.open(input_tiff_path)

    # The top left corner of tiff raster
    x_start = input_tiff_raster.transform[2]
    y_start = input_tiff_raster.transform[5]
    print(x_start, y_start)

    # Metes covered in each direction for given tile size
    x_step = input_tiff_raster.transform[0] * tile_size
    y_step = input_tiff_raster.transform[4] * tile_size
    print(x_step, y_step)
    
    # Read the RGB image from tiff raster (C, H, W) format
    tiff_image = input_tiff_raster.read()

    # Find the number of tiles in each direction to create
    _, height, width = tiff_image.shape
    num_rows = math.ceil(height / tile_size)
    num_cols = math.ceil(width / tile_size)

    # Create the csv file with the same name as tiff file
    tiles_csv_out_path = os.path.join(tiles_csv_out_dir, folder_name+'.csv')
    fw = open(tiles_csv_out_path, 'w')
    fw.write("AOI,id,Geometry\n")
    fw.close()
    
    total_tiles = num_rows * num_cols
    print("Total {} tiles for the given image {}".format(total_tiles, tiff_file))

    # To maintain unique file name
    idx = 0

    # loop through each row (x direction)
    for row in range(num_rows):
        # loop through each column of that row
        for col in range(num_cols):
            idx += 1

            # Find the positions this current tile based on row and col number
            x_min_geo = float(x_start + col * x_step)         # left edge
            x_max_geo = float(x_min_geo + x_step)             # right edge
            y_max_geo = float(y_start + row * y_step)         # bottom edge
            y_min_geo = float(y_max_geo + y_step)             # top edge
            
            # Create polygon geometry for the tile
            polygon_str = "{} {},{} {},{} {},{} {},{} {}".format(x_min_geo, y_min_geo, x_max_geo, y_min_geo,
                                                            x_max_geo, y_max_geo, x_min_geo, y_max_geo,
                                                            x_min_geo, y_min_geo)

            # Write the polygon geometry in the csv file with tile idx and tiff file name
            fa = open(tiles_csv_out_path, 'a')
            fa.write("%s,%d,\"POLYGON ((%s))\"\n" % ("{}".format(tiff_file.split('.')[0]), idx, polygon_str))
            fa.close()

            # Capture the data from large tiff file for current tile
            window = Window(col * tile_size, row * tile_size, tile_size, tile_size)
            data = input_tiff_raster.read(window=window)

            # Check if padding is required for the current tile
            if data.shape != expected_tile_size:
                # Create a padded array with zeros
                padded_data = np.zeros(expected_tile_size, dtype=data.dtype)

                # Calculate the dimensions for copying the data
                copy_height = min(tile_size, height - row * tile_size)
                copy_width = min(tile_size, width - col * tile_size)

                # Copy the data to the padded array
                padded_data[:, :copy_height, :copy_width] = data[:, :copy_height, :copy_width]

                # Use the padded array as the data for writing
                data = padded_data

            # The name of tiff file for current tile
            output_tiff_file = f"{idx}.tif"
            output_tiff_file_path = os.path.join(tiles_tif_out_dir, output_tiff_file)

            # Find the geo-coord transformation for current tile
            tile_transform = from_origin(x_min_geo, y_max_geo,
                                         input_tiff_raster.transform[0], -1*input_tiff_raster.transform[4])
            
            # print(tiff_raster.crs)

            # Save the tiff file at desired location
            with rasterio.open(output_tiff_file_path, 'w', driver='GTiff',
                               width=expected_tile_size[2], height=expected_tile_size[1],
                               count=input_tiff_raster.count, dtype=data.dtype,
                               crs=crs.to_wkt(), transform=tile_transform) as dst:
                dst.write(data)

    # break