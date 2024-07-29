import os
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from qgis.core import *
import math

home_path = QgsProject.instance().homePath()
read_dir = os.path.join(home_path,"RoadTracer_Coordinates")
output_dir = os.path.join(home_path,"Outputs")
file_list = os.listdir(read_dir)
page_x, page_y = 346.8,346.8
project = QgsProject.instance()
manager = project.layoutManager()


for file in file_list:
    folder_name = os.path.join("RoadTracer",file.split('.')[0])
    f = open(os.path.join(read_dir,file))
    for line in f.readlines():
        data = line.split(",")
        if len(data) == 5:
            file_name = data[0]
            lng1 = float(data[1])
            lat1 = float(data[2])
            lng2 = float(data[3])
            lat2 = float(data[4])
            p1 = QgsPointXY((float(lng1)), float(lat2))
            p2 = QgsPointXY((float(lng2)), float(lat1))
            extent = QgsRectangle (p1,p2)
            
            layouts_list = manager.printLayouts()
            # remove any duplicate layouts
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
            map.setCrs(QgsCoordinateReferenceSystem("EPSG:900913"))
            map.setExtent(extent)
            map.setBackgroundColor(QColor(255, 255, 255, 0))
            
            layout.addLayoutItem(map)
            layout = manager.layoutByName("MyLayout")
            exporter = QgsLayoutExporter(layout)
            
            folder_path = os.path.join(output_dir,folder_name)
            print(folder_path)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            img_path = os.path.join(folder_path,file_name)
            
            settings = QgsLayoutExporter.ImageExportSettings()
            exporter.exportToImage(img_path, settings)
            
'''
#settings.imageSize = QSize(826, 1169)
#settings.dpi = 292
'''