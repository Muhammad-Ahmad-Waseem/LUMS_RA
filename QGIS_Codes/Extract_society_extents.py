import os
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from qgis.core import *

home_path = QgsProject.instance().homePath()
file_path = "D:\LUMS_RA\Zameen Data\Statistics.csv"
output_dir = "D:\LUMS_RA\Google_Earth_Images_Downloader"

f = open(file_path)
lines = f.readlines()
f.close() 

total_images = len(lines)-1
print("Starting for {} Socities".format(total_images))

layer = QgsProject.instance().mapLayersByName("1_preds")[0]
counter = 0
for line in lines:
    society_wths = line.split(",")[0]
    if(society_wths == "Society"):
        continue
    layer.removeSelection()
    layer.selectByExpression('"society-na" like \'{}%\''.format(society_wths), QgsVectorLayer.SetSelection)
    print(society_wths)
    selection = layer.selectedFeatures()
    caps = layer.dataProvider().capabilities()
    #break
    for feature in selection:
        #print(feature.attributes())
        if caps & QgsVectorDataProvider.ChangeAttributeValues:
            print("here:")
            attrs = { 0 : society_wths, 1 :  feature.attributes()[1]}
            layer.dataProvider().changeAttributeValues({ feature.id() : attrs })
    print(counter)
    counter = counter+1
    continue
    box = layer.boundingBoxOfSelected()
    
    soc_wo_space = society_wths.split(" ")
    society_name = soc_wo_space[0]
    
    for i in range(len(soc_wo_space)-1):
        society_name = society_name +"_"+ soc_wo_space[i+1]
    
    folder_path = os.path.join(output_dir,society_name)
        
    if(os.path.exists(folder_path)):
        continue
        
    os.makedirs(folder_path)
    f = open(os.path.join(folder_path,"Extent.txt"),"w+")
    f.write("Long_min:{}\n".format(box.xMinimum()))
    f.write("Long_max:{}\n".format(box.xMaximum()))
    f.write("Lat_min:{}\n".format(box.yMinimum()))
    f.write("Lat_max:{}\n".format(box.yMaximum()))
    f.close()
    
    counter = counter+1
    print("Done {}!".format(counter))
    #break