from PyQt5.QtCore import *
from PyQt5.QtGui import *
from qgis.core import *

crsSrc = QgsCoordinateReferenceSystem("EPSG:4326")    # WGS 84
crsDest = QgsCoordinateReferenceSystem("EPSG:4326")  # WGS 84 / UTM zone 33N
transformContext = QgsProject.instance().transformContext()
xform = QgsCoordinateTransform(crsSrc, crsDest, transformContext)

extent = QgsRectangle(xform.transform(QgsPointXY (74.334195029,31.505383102)),xform.transform(QgsPointXY (74.400198828,31.538597917)))
canvas = iface.mapCanvas()
canvas.setExtent(extent)
canvas.refresh()