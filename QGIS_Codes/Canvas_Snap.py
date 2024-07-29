from PyQt5.QtCore import *
from PyQt5.QtGui import *
from qgis.core import *
import os
#Make sure EPSG:4326 is set.

canvas= iface.mapCanvas()
project = QgsProject.instance()
manager = project.layoutManager()
home_path = QgsProject.instance().homePath()
ext_layer = QgsProject.instance().mapLayersByName('Areas')[0]
extent = ext_layer.extent()
for i in range(16):
    layouts_list = manager.printLayouts()
    for layout in layouts_list:
            if layout.name() == "Layout":
                manager.removeLayout(layout)
    layer = QgsProject.instance().mapLayersByName('out_{}'.format(i+1))[0]
    raster = QgsProject.instance().mapLayersByName('Cliped_{}'.format(i+1))[0]
    #project.layerTreeRoot().findLayer(layer.id()).setItemVisibilityChecked(True)
    project.layerTreeRoot().findLayer(raster.id()).setItemVisibilityChecked(True)

    layout = QgsPrintLayout(project)
    layout.initializeDefaults()
    layout.setName("Layout")
    manager.addLayout(layout)

    # create map item in the layout
    map = QgsLayoutItemMap(layout)
    map.setRect(0,0,20,20)

    map.zoomToExtent((extent))
    
    #map.zoomToExtent(QgsRectangle (float(74.454),float(31.422),float(74.515),float(31.521)))
    map.attemptResize(QgsLayoutSize(295, 210, QgsUnitTypes.LayoutMillimeters))
    map.setBackgroundColor(QColor(255, 255, 255, 0))
    layout.addLayoutItem(map)

    layout = manager.layoutByName("Layout")
    exporter = QgsLayoutExporter(layout)

    image_name = "{}.png".format(i+1)
    image_path = os.path.join(home_path,image_name)

    settings = QgsLayoutExporter.ImageExportSettings()
    exporter.exportToImage(image_path, settings)
    #project.layerTreeRoot().findLayer(layer.id()).setItemVisibilityChecked(False)
    project.layerTreeRoot().findLayer(raster.id()).setItemVisibilityChecked(False)