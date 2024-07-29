import processing
import subprocess
import glob
import os
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from qgis.core import *
import math

LST_layer = 'D://LUMS_RA//Dr Aamir//SDSC Paper//Data/2021//LST_2021.tif'
Layers = 'D://LUMS_RA//Dr Aamir//New Paper//Green Spaces//Separated_Layers'
out_folder = "D://LUMS_RA//Dr Aamir//New Paper//Green Spaces//output_layers"
inter_folder = "D://LUMS_RA//Dr Aamir//New Paper//Green Spaces//Intermediates"
files = [file for file in os.listdir(Layers) if file.endswith('.shp')]
print(len(files))
for file in files:
    INIT_layer = os.path.join(Layers, file)
    temp_folder = os.path.join(inter_folder, file)
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
        
    distance = 30
    points = 35

    for i in range(points+1):
        if i == 0:
            processing.run("native:zonalstatisticsfb",
            {'INPUT':INIT_layer,
            'INPUT_RASTER':LST_layer,
            'RASTER_BAND':1,
            'COLUMN_PREFIX':'_',
            'STATISTICS':[2],
            'OUTPUT':os.path.join(out_folder, 'Zonal_stats_{}_{}.csv'.format(file, distance*i))})
        else:
            processing.run("native:buffer",
            {'INPUT':INIT_layer,
            'DISTANCE':15,
            'SEGMENTS':5,
            'END_CAP_STYLE':0,
            'JOIN_STYLE':0,
            'MITER_LIMIT':2,
            'DISSOLVE':False,
            'OUTPUT':os.path.join(temp_folder, 'Inter1_{}.shp'.format(i))})
            
            processing.run("native:difference",
            {'INPUT':os.path.join(temp_folder, 'Inter1_{}.shp'.format(i)),
            'OVERLAY':INIT_layer,
            'OUTPUT':os.path.join(temp_folder, 'Inter2_{}.shp'.format(i))})
            
            processing.run("native:zonalstatisticsfb",
            {'INPUT':os.path.join(temp_folder, 'Inter2_{}.shp'.format(i)),
            'INPUT_RASTER':LST_layer,
            'RASTER_BAND':1,
            'COLUMN_PREFIX':'_',
            'STATISTICS':[2],
            'OUTPUT':os.path.join(out_folder, 'Zonal_stats_{}_{}.csv'.format(file, distance*i))})
            
            INIT_layer = os.path.join(temp_folder, 'Inter1_{}.shp'.format(i))
    print("File: {} done".format(file))
            
            