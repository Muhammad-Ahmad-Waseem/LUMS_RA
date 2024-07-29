from qgis.core import *
import math

zoom_level = 20
def zoom_to_scale(zoom_level):
    dpi=iface.mainWindow().physicalDpiX()
    maxScalePerPixel = 156543.04
    inchesPerMeter = 39.37
    scale = (dpi* inchesPerMeter * maxScalePerPixel) / (2**zoom_level)
    return scale

canvas= iface.mapCanvas()
canvas.zoomScale(zoom_to_scale(zoom_level))
canvas.refresh()