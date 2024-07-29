import os
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from qgis.core import *
import math

def zoom_to_scale(zoom_level):
    dpi=iface.mainWindow().physicalDpiX()
    maxScalePerPixel = 156543.04
    inchesPerMeter = 39.37
    scale = (dpi* inchesPerMeter * maxScalePerPixel) / (2**zoom_level)
    return scale

f = open("Lahore_map_list1.txt")
c = np.array(['','',''])

for line in f.readlines():
    parts = line.split("    ")
    if len(parts) == 5:
        #print(len(parts))
        fName = parts[0]
        long_left = float(parts[1])
        long_right = float(parts[2])
        lat_top = float(parts[3])
        lat_bottom = float(parts[4].strip('\n'))
        Long_center = (long_left + long_right)/2
        Lat_center = (lat_top + lat_bottom)/2
        c = np.vstack((c, [fName, Long_center, Lat_center]))

#delete first row
c = np.delete(c, (0), axis=0)
print(c)
allData = c

zoom_levels = [14,15,16,17,18]
canvas= iface.mapCanvas()
layer = iface.activeLayer()
extent = layer.extent()
project = QgsProject.instance()
manager = project.layoutManager()

mainPath = 'Lahore_Images//'
imageType = "png"

crsSrc = QgsCoordinateReferenceSystem("EPSG:4326")    # WGS 84
crsDest = QgsCoordinateReferenceSystem("EPSG:3857")  # WGS 84 / UTM zone 33N
transformContext = QgsProject.instance().transformContext()
xform = QgsCoordinateTransform(crsSrc, crsDest, transformContext)

base_path = os.path.join(QgsProject.instance().homePath(),mainPath)
if not os.path.exists(base_path):
        os.makedirs(base_path)
        
if not os.path.exists(os.path.join(base_path,"List.txt")):
    write_file = os.path.join(base_path,"List.txt")
    f = open(write_file, "w")
    f.write('Image_Name\tLongitude_Center\tLatitude_Center\tZoom_Level\n')
    f.close()
else:
    print("File found, appending current data into it")

for zoom_level in zoom_levels:
    for i in range(0, allData.shape[0]):
        info = allData[i][0].split('/')
        if int(info[0]) == 13:
            #zoom_level = int(info[0])
            #filename = allData[i][1]+ "_" + allData[i][2]
            filename = 'Level{}_Img{}_Mask'.format(zoom_level,i+1) + '.' + imageType
            TranPoint = xform.transform(QgsPointXY(float(allData[i][1]),float(allData[i][2])))
            canvas.setExtent(extent)
            canvas.zoomScale(zoom_to_scale(zoom_level))
            canvas.setCenter(QgsPointXY((float(TranPoint[0])), float(TranPoint[1])))
                
            layouts_list = manager.printLayouts()

            # remove any duplicate layouts
            for layout in layouts_list:
                if layout.name() == "MyLayout":
                    manager.removeLayout(layout)
            
            layout = QgsPrintLayout(project)
            layout.initializeDefaults()
            layout.setName("MyLayout")
            manager.addLayout(layout)

            # create map item in the layout
            map = QgsLayoutItemMap(layout)
            map.setRect(20, 20, 20, 20)
            
            map.zoomToExtent(iface.mapCanvas().extent())
            map.setBackgroundColor(QColor(255, 255, 255, 0))

            map.attemptMove(QgsLayoutPoint(5, 5, QgsUnitTypes.LayoutMillimeters))
            map.attemptResize(QgsLayoutSize(285, 200, QgsUnitTypes.LayoutMillimeters))

            layout.addLayoutItem(map)

            layout = manager.layoutByName("MyLayout")
            exporter = QgsLayoutExporter(layout)
            
            #image_path = base_path + filename + '.' + imageType
            image_path = base_path + filename

            exporter.exportToImage(image_path, QgsLayoutExporter.ImageExportSettings())
            f = open(write_file, "a")
            f.write('{}\t{}\t{}\t{}\n'.format(filename,allData[i][1],allData[i][2],zoom_level))
            f.close()