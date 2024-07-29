import os
import subprocess
import shlex

# Define input and output files
outRasterDir =  "D://LUMS_RA//QGIS//Outputs//Area//Rasters"
inpImageDir =  "D://LUMS_RA//QGIS//Outputs//Area//Images"
gcp_poits = "D://LUMS_RA//QGIS//area_coords.txt"
img_x = 1024
img_y = 1024

if not os.path.exists(outRasterDir):
    os.makedirs(outRasterDir)
    
temp_file_dir = os.path.join(outRasterDir,"tmp")
if not os.path.exists(temp_file_dir):
    os.makedirs(temp_file_dir)

    
f = open(gcp_poits)
for line in f.readlines():
    data = line.strip().split(",")
    if data[0].endswith(".png"):
        spl = data[0].split('.')
        sp2 = spl[0].split("_")
        raster = sp2[1] + ".tif"
        img = sp2[1]+".png"
        inputfile = os.path.join(inpImageDir,img)
        tmp_file = os.path.join(temp_file_dir,img)
        outputfile = os.path.join(outRasterDir,raster)

        coords = [data[1],data[3],data[4],data[2]]
        gcp1 = "{} {} {} {}".format(0,0,float(coords[0]),float(coords[2]))
        gcp2 = "{} {} {} {}".format(img_x,0,float(coords[1]),float(coords[2]))
        gcp3 = "{} {} {} {}".format(img_x,img_y,float(coords[1]),float(coords[3]))
        gcp4 = "{} {} {} {}".format(0,img_y,float(coords[0]),float(coords[3]))

        command1 = "gdal_translate -of GTiff -gcp {} -gcp {} -gcp {} -gcp {} {} {}".format(gcp1,gcp2,gcp3,gcp4,inputfile,tmp_file)
        command2 = "gdalwarp -r near -order 1 -co COMPRESS=NONE  -t_srs EPSG:3857 {} {}".format(tmp_file,outputfile)
        
        #print(command1)
        
        if (os.system(command1)==0):
            print("Translation done for {}".format(img))
        if (os.system(command2)==0):
            print("Raster {} created!".format(img))

        #break
        '''
        
        fw.write('{},{},0,-,1\n'.format(float(data[1]),float(data[4])))
        fw.write('{},{},4096,-4096,1\n'.format(float(data[3]),float(data[4])))
        fw.write('{},{},4096,0,1\n'.format(float(data[3]),float(data[2])))
        fw.write('{},{},0,0,1\n'.format(float(data[1]),float(data[2])))
        fw.close()'''
# Generate the command
#command = "gdal_translate -of GTiff -gcp {} {} {} {} -gcp 1631.04 846.306 12323 1232 -gcp 223.864 899.722 2.13241e+08 132314 -gcp 1694.47 191.963 1.24312e+07 14314 {} {}".format(1000,1000,10,10,inputfile,outputfile)
#print(command)

# Run the command. os.system() returns value zero if the command was executed succesfully
#print(os.system(command))
