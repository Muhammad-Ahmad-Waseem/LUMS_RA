import processing
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from qgis.core import *
import os
import subprocess
import glob

input_images = "D:\\LUMS_RA\\Predictions\\Current_Model\\Segmentations\\Complete_Zameen\\**\*.tif"
imgs = [file for file in glob.glob(input_images)]
for img in imgs:
    file = os.path.split(img)
    file_path = file[0]
    file_name = file[1].split('.')[0]
    
    #temp_file1 = os.path.join(file_path, file_name+'_rgb2pct.tif')
    #temp_file2 = os.path.join(file_path, file_name+'_vctrizd.shp')
    #temp_file3 = os.path.join(file_path, file_name+'_filterd.shp')
    out_file = os.path.join(file_path, file_name+'_vctrizd.shp')
 
    '''processing.run("gdal:rgbtopct", 
                    { 
                    'INPUT' : img, 
                    'NCOLORS' : 2,
                    'OUTPUT' : temp_file1})
    '''
    processing.run("gdal:polygonize", 
                    { 
                    'INPUT' : img, 
                    'BAND' : 1,
                    'FIELD' : 'DN',
                    'EIGHT_CONNECTEDNESS':False,
                    'EXTRA':'',
                    'OUTPUT' : out_file})
    '''                
    processing.run("native:extractbyattribute", 
                    { 
                    'INPUT' : temp_file2, 
                    'FIELD' : 'DN',
                    'OPERATOR':0,
                    'VALUE':'1',
                    'OUTPUT' : temp_file3})
    
    processing.run("native:fixgeometries", 
                    { 
                    'INPUT' : temp_file3,
                    'OUTPUT' : out_file})'''
                    
    #subprocess.check_output("del /f {}".format(img), shell=True)
    #subprocess.check_output("del /f {}".format(temp_file2), shell=True)
    #subprocess.check_output("del /f {}".format(temp_file3), shell=True)
    break
    