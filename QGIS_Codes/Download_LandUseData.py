'''
Before Running this code, just make sure you hide all latyers from project QGIS.
This code will automatically pick defined layers and hide/unhide them propoerly.
Make sure you set the colors of polygons according to your need, before running
this code.
'''

import os
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from qgis.core import *
import math

home_path = QgsProject.instance().homePath()
file_path = "D:\\LUMS_RA\\Marking_Areas (Non DHA)\\Area5_Hamza\\coords.txt"
output_dir = "D:\\LUMS_RA\\Marking_Areas (Non DHA)\\Area5_Hamza\\QGIS_Dataset"
page_x, page_y = 21.675,21.675
project = QgsProject.instance()
manager = project.layoutManager()
'''
raster_layers = ['22-10-2020', '18-5-2020',
'6-2-2020','22-2-2019','21-11-2018',
'1-5-2018','19-4-2017','11-4-2016',
'22-6-2015','9-11-2014','18-3-2014',
'5-12-2013','6-5-2013','27-11-2012',
'3-5-2012','4-5-2011','2-2-2010']

mask_layers = ['22_10_2020_GT', '18_5_2020_GT',
'6_2_2020_GT','22_2_2019_GT','21_11_2018_GT',
'1_5_2018_GT','19_4_2017_GT','11_4_2016_GT',
'22_6_2015_GT','9_11_2014_GT','18_3_2014_GT',
'5_12_2013_GT','6_5_2013_GT','27_11_2012_GT',
'3_5_2012_GT','4_5_2011_GT','2_2_2010_GT']'''

#for i in range(len(raster_layers)):    
    #folder_name = os.path.join("DHA_GEID_Temporal2",raster_layers[i])
'''folder_name = "DHA_GEID_Current2"
sub_folder1 = os.path.join(folder_name,"Images")
sub_folder2 = os.path.join(folder_name,"Masks")
sub_folder3 = os.path.join(folder_name,"Contours")'''

#Define Each sub_folder you need to create
sub_folders = ['Images','Masks']

#Define Layer Names for each of these sub folders
layer_names = ['area_combined_image','vectorized_layer']

#Define background color for each sub class
colors = [QColor(0, 0, 0, 255),QColor(255, 255, 255, 255)]

#Define if you need to mask any layer based on other
Masks = [True,False]

#Define mask layer
mask_layer = 'congested_areas'

f = open(file_path)
lines = f.readlines()
f.close() 

total_images = len(lines)-1

assert len(sub_folders) == len(layer_names),"Sub_Folder and layer names must have same size"
assert len(sub_folders) == len(colors),"Sub_Folder and colors must have same size"
assert len(sub_folders) == len(Masks),"Sub_Folder and masks array must have same size"
assert len(layer_names) == len(colors),"Layer names and colors array must have same size"
assert len(layer_names) == len(Masks),"Layer names and masks array must have same size"
assert len(colors) == len(Masks),"Colors and masks array must have same size"
assert total_images > 0, "Total images must be greater than zero"

print("Starting download {} for {} locations".format(sub_folders,total_images))

counter = 0
for line in lines:
    data = line.split(",")
    if len(data) == 5 and "." in data[0]:# and counter==35:
        counter +=1
        file_name = data[0]
        extent = QgsRectangle (float(data[1]),float(data[2]),float(data[3]),float(data[4]))
        for i,folder in enumerate(sub_folders):
            layer = QgsProject.instance().mapLayersByName(layer_names[i])[0]
            masklayer = None
            if(Masks[i]):
                masklayer = QgsProject.instance().mapLayersByName(mask_layer)[0]
            project.layerTreeRoot().findLayer(layer.id()).setItemVisibilityChecked(True)
            
            if masklayer is not None:
                project.layerTreeRoot().findLayer(masklayer.id()).setItemVisibilityChecked(True)
                
            layouts_list = manager.printLayouts()
            for layout in layouts_list:
                if layout.name() == "MyLayout":
                    manager.removeLayout(layout)
            layout = QgsPrintLayout(project)
            layout.initializeDefaults()
            layout.setName("MyLayout")
            pc = layout.pageCollection()
            pc.pages()[0].setPageSize(QgsLayoutSize(page_x, page_y, QgsUnitTypes.LayoutMillimeters))
            manager.addLayout(layout)
            map = QgsLayoutItemMap(layout)
            map.setRect(0,0,page_x,page_y)
            map.setCrs(QgsCoordinateReferenceSystem("EPSG:3857"))
            map.setExtent(extent)
            map.setBackgroundColor(colors[i])

            layout.addLayoutItem(map)
            layout = manager.layoutByName("MyLayout")
            exporter = QgsLayoutExporter(layout)

            folder_path = os.path.join(output_dir,folder)
            #print(folder_path)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                
            img_path = os.path.join(folder_path,file_name)
            settings = QgsLayoutExporter.ImageExportSettings()
            exporter.exportToImage(img_path, settings)
            project.layerTreeRoot().findLayer(layer.id()).setItemVisibilityChecked(False)
            if masklayer is not None:
                project.layerTreeRoot().findLayer(masklayer.id()).setItemVisibilityChecked(False)
            
        print("File {} of {} done!".format(counter,total_images))