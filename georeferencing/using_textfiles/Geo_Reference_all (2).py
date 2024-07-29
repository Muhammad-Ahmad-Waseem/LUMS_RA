import os
from tqdm import tqdm
import sys
import subprocess

main_dir = os.getcwd()
dirs = [dir for dir in os.listdir(main_dir) if not os.path.isfile(dir)]

#Define positions of coordinates in file
xmin = 0
xmax = 1
ymax = 2
ymin = 3

for dir in dirs:
    if dir != "Time5":# and dir != "Time1":
        # Define input and output files
        outRasterDir =  os.path.join(dir,"rasters")
        inpImageDir =  os.path.join(dir,"pred3")
        gcp_poits = os.path.join(dir,"Images_list1.txt")
        img_x = 256
        img_y = 256

        if not os.path.exists(outRasterDir):
            os.makedirs(outRasterDir)
            
        temp_file_dir = os.path.join(outRasterDir,"tmp")
        if not os.path.exists(temp_file_dir):
            os.makedirs(temp_file_dir)

            
        f = open(gcp_poits)
        lines= f.readlines()
        for line in tqdm(lines,desc="{}".format(dir), file=sys.stdout):
            data = line.strip().split(":")
            if data[0].endswith(".jpg"):
                spl = data[0].split('.')
                raster = spl[0] + ".tif"
                img = spl[0]+".jpg"
                inputfile = os.path.join(inpImageDir,img)
                tmp_file = os.path.join(temp_file_dir,img)
                outputfile = os.path.join(outRasterDir,raster)

                coords = data[1].strip().split("    ")
                gcp1 = "{} {} {} {}".format(0,0,float(coords[xmin]),float(coords[ymax]))
                gcp2 = "{} {} {} {}".format(img_x,0,float(coords[xmax]),float(coords[ymax]))
                gcp3 = "{} {} {} {}".format(img_x,img_y,float(coords[xmax]),float(coords[ymin]))
                gcp4 = "{} {} {} {}".format(0,img_y,float(coords[xmin]),float(coords[ymin]))

                command1 = "gdal_translate -of GTiff -gcp {} -gcp {} -gcp {} -gcp {} {} {}".format(gcp1,gcp2,gcp3,gcp4,inputfile,tmp_file)
                command2 = "gdalwarp -r near -order 1 -co COMPRESS=NONE  -t_srs EPSG:4326 {} {}".format(tmp_file,outputfile)
                
                subprocess.call(command1,shell=True)
                subprocess.call(command2,shell=True)
                #break
                '''
                if (os.system(command1)!=0):
                    print("Translation failed for {}".format(img))
                if (os.system(command2)!=0):
                    print("Raster creattion failed for {}".format(img))'''
            print("")

    #os.remove(temp_file_dir)
