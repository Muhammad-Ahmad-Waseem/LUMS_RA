import os
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from qgis.core import *
import math

home_path = QgsProject.instance().homePath()
path_to_coords = "D:\\LUMS_RA\\Marking_Areas (Non DHA)\\Area5_Hamza\\coords.txt"

crs1 = QgsCoordinateReferenceSystem("EPSG:4326")    # WGS 84
crs2 = QgsCoordinateReferenceSystem("EPSG:3857")  # WGS 84 / UTM zone 33N

transformContext = QgsProject.instance().transformContext()
xform1 = QgsCoordinateTransform(crs1, crs2, transformContext)
#xform2 = QgsCoordinateTransform(crs2, crs1, transformContext)

#layers = [layer.name() for layer in QgsProject.instance().mapLayers().values()]

Area_layer = QgsProject.instance().mapLayersByName("Area5")[0]
#mask_layer = QgsProject.instance().mapLayersByName("4_10_2021_GT")[0]

extent = xform1.transform(Area_layer.extent()) 
print(extent)
xmin = extent.xMinimum()
xmax = extent.xMaximum()
ymin = extent.yMinimum()
ymax = extent.yMaximum()

print(extent)
grid_x = 300 #meters
grid_y = 300 #meters

xidx = xmin

count = 0
f = open(path_to_coords,"w+")
f.write("Image_Name,xmin,ymin,xmax,ymax\n")
f.close()

while(xidx < xmax):
    yidx = ymin
    while(yidx < ymax):
        count += 1
        f = open(path_to_coords,"a+")
        f.write("{}.png,{},{},{},{}\n".format(count,xidx,yidx,xidx+grid_x,yidx+grid_y))
        f.close()
        yidx += grid_y
    xidx += grid_x
    '''
    while(yidx < ymax):
        extent = QgsRectangle(xidx,yidx,xidx+grid_x,yidx+grid_y)
        request = QgsFeatureRequest().setFilterRect(xform2.transform(extent))
        features =  [feature for feature in mask_layer.getFeatures(request)]

        if len(features) != 0:
            #print(extent)
            count += 1
            f = open(os.path.join(home_path,"Coordinates_Temp.txt"),"a+")
            f.write("{}.png,{},{},{},{}\n".format(count,xidx,yidx,xidx+grid_x,yidx+grid_y))
            f.close()
            
        yidx += grid_y
    xidx += grid_x
    '''

print("Coordinates of {} Images added".format(count))

'''
field_names = [field.name() for field in layer.fields()]
features = layer.getFeatures()
for feature in features:
    print((feature.attributes()))
    #if(feature.hasGeometry()):
        #print(feature.geometry())
    break;
#areaOfInterest = QgsRectangle(450290,400520, 450750,400780)

request = QgsFeatureRequest().setFilterRect(extent)
features =  [feature for feature in layer.getFeatures(request)]

print(len(features))
'''