from qgis.core import *
import os
import processing
from processing.core.Processing import Processing

qgs = QgsApplication([], False)
QgsApplication.setPrefixPath("C://Program Files//QGIS 3.16.9", True)
qgs.initQgis()

Processing.initialize()
home_path = QgsProject.instance().homePath()
WP_data_dir = "D://LUMS_RA//Population Data//pak_ppp_2020_1km_Aggregated.tif"
Districts_dir = "D:\LUMS_RA\Shape Files Pak\Pakistan Districts,Unions\District_Boundary.shp"
Out_dir = "D://LUMS_RA//New Hiring Training"

if not os.path.exists(Out_dir):
    os.makedirs(Out_dir)

fixed_geoms = os.path.join(Out_dir,"fixed_geom_dist.shp")


processing.run("native:fixgeometries",
{'INPUT':Districts_dir,
'OUTPUT':fixed_geoms})

out = os.path.join(Out_dir,"zonal_stats.csv")

parameters = {'INPUT': fixed_geoms,
'INPUT_RASTER':WP_data_dir,
'RASTER_BAND':1,
'COLUMN_PREFIX':'_',
'STATISTICS':[0,1,2],
'OUTPUT':out}
processing.run("native:zonalstatisticsfb",parameters)