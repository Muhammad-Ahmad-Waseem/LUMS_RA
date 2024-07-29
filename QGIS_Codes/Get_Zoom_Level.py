import math
scale=iface.mapCanvas().scale()
dpi=iface.mainWindow().physicalDpiX()
maxScalePerPixel = 156543.04
inchesPerMeter = 39.37
zoomlevel = int(round(math.log( ((dpi* inchesPerMeter * maxScalePerPixel) / scale), 2 ), 0))
print (zoomlevel)