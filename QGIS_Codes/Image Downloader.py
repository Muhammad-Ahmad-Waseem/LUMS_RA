import math
zoom_level = 20
dpi=iface.mainWindow().physicalDpiX()
maxScalePerPixel = 156543.04
inchesPerMeter = 39.37
scale = (dpi* inchesPerMeter * maxScalePerPixel) / (2**zoom_level)

map_settings = iface.mapCanvas().mapSettings()
layer = iface.activeLayer()
extent = layer.extent()

crsSrc = QgsCoordinateReferenceSystem("EPSG:4326")    # WGS 84
crsDest = QgsCoordinateReferenceSystem("EPSG:3857")  # WGS 84 / UTM zone 33N
transformContext = QgsProject.instance().transformContext()
xform = QgsCoordinateTransform(crsSrc, crsDest, transformContext)

Long_center = (74.091796875 + 74.1796875)/2
Lat_center = (31.65338139677498 + 31.57853542358965)/2
TranPoint = xform.transform(QgsPointXY(float(Long_center),float(Lat_center)))
canvas = iface.mapCanvas()
canvas.setExtent(extent)

canvas.zoomScale(scale)
canvas.setCenter(QgsPointXY((float(TranPoint[0])), float(TranPoint[1])))
canvas.refresh()


project = QgsProject.instance()
manager = project.layoutManager()
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



base_path = os.path.join(QgsProject.instance().homePath())
image_path = os.path.join(base_path, "output.png")

exporter.exportToImage(image_path, QgsLayoutExporter.ImageExportSettings())