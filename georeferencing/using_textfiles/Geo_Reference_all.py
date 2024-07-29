import os
from tqdm import tqdm
import sys
import subprocess

main_dir = "D:\\LUMS_RA\\Temp_Work\\testing_salman"
temp_dir = os.path.join(main_dir, "tmp")
RasterDir = os.path.join(main_dir, "rasters")

#Define Subfolder here: "images" or "predictions"
#subfolder = "predictions"
#dirs = [dir for dir in os.listdir(main_dir) if not os.path.isfile(dir)]

#for dir in dirs:
#Define path for output (save it in main dir for easier access)
outRasterDir = RasterDir#os.path.join(RasterDir, dir)
temp_file_dir = temp_dir#os.path.join(te    mp_dir, dir)

#Define input paths, placed in subdirectories
dir_path = main_dir#os.path.join(main_dir, dir)
inpImageDir = dir_path
#inpImageDir = os.path.join(inpImageDir, subfolder)
gcp_ponits = "D:\\LUMS_RA\\Temp_Work\\lahore_salman.txt"#os.path.join(dir_path, "corrds.txt")

if not os.path.exists(outRasterDir):
    os.makedirs(outRasterDir)
if not os.path.exists(temp_file_dir):
    os.makedirs(temp_file_dir)

#Define positions of coordinates in file
xmin = 0
xmax = 2
ymax = 1
ymin = 3
img_x = 4096
img_y = 4096

f = open(gcp_ponits)
lines= f.readlines()
for line in tqdm(lines,desc="{}".format(dir), file=sys.stdout):
    data = line.strip().split(",")
    if data[0].endswith(".png"):
        spl = data[0].split('.')
        raster = spl[0] + ".tif"
        img = spl[0]+".png"
        inputfile = os.path.join(inpImageDir, img)
        tmp_file = os.path.join(temp_file_dir,img)
        outputfile = os.path.join(outRasterDir,raster)

        coords = data[1:]#data[1].strip().split("    ")
        width = img_x#coords[img_x]
        height = img_y#coords[img_y]
        gcp1 = "{} {} {} {}".format(0, 0,float(coords[xmin]),float(coords[ymax]))
        gcp2 = "{} {} {} {}".format(width,0,float(coords[xmax]),float(coords[ymax]))
        gcp3 = "{} {} {} {}".format(width,height,float(coords[xmax]),float(coords[ymin]))
        gcp4 = "{} {} {} {}".format(0,height,float(coords[xmin]),float(coords[ymin]))

        command1 = "gdal_translate -of GTiff -gcp {} -gcp {} -gcp {} -gcp {} {} {}".format(gcp1,gcp2,gcp3,gcp4,inputfile,tmp_file)
        command2 = "gdalwarp -r near -order 1 -co COMPRESS=NONE  -t_srs EPSG:3857 {} {}".format(tmp_file,outputfile)

        subprocess.call(command1,shell=True)
        subprocess.call(command2,shell=True)
        subprocess.check_output("del /f {}".format(tmp_file), shell=True)
        #break
    print("")
