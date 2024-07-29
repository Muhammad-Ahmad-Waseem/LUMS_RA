import os
from tqdm import tqdm
import sys
import subprocess

main_dir = os.getcwd()
dirs = [dir for dir in os.listdir(main_dir) if not os.path.isfile(dir)]

for dir in tqdm(dirs,file=sys.stdout):
    filesDir = os.path.join(main_dir,dir)
    vrtDir = os.path.join(main_dir,"mosaics")

    if not os.path.exists(vrtDir):
        os.makedirs(vrtDir)
    
    vrtFileDir = os.path.join(vrtDir,"{}.vrt".format(dir))
    rasterFilesDir = os.path.join(filesDir,"rasters\\*.tif")
    outRasterDir = os.path.join(filesDir,"{}.tif".format(dir))
    
    command1 = "gdalbuildvrt {} {}".format(vrtFileDir,rasterFilesDir)
    command2 = 'gdal_translate -of GTiff -co "COMPRESS=JPEG" -co "PHOTOMETRIC=YCBCR" -co "TILED=YES" {} {}'.format(vrtFileDir,outRasterDir)
    
    subprocess.call(command1,shell=True)
    subprocess.call(command2,shell=True)
    #break
    #print(command1)
