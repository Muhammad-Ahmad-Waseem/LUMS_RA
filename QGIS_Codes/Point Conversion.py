import os
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from qgis.core import *
import math

#point1 lng = -7922870.281284 , lat = 5221401.062440
#point2 lng = -7920424.296379 , lat = 5218955.077535
crsSrc = QgsCoordinateReferenceSystem("EPSG:900913")    # WGS 84
crsDest = QgsCoordinateReferenceSystem("EPSG:4326")  # WGS 84 / UTM zone 33N
transformContext = QgsProject.instance().transformContext()
xform = QgsCoordinateTransform(crsSrc, crsDest, transformContext)

p1 = QgsPointXY(float(-7922870.281284),float(5221401.062440))
p2 = QgsPointXY(float(-7920424.296379),float(5218955.077535))

TranPoint1 = xform.transform(p1)
TranPoint2 = xform.transform(p2)

print(TranPoint1)
print(TranPoint2)