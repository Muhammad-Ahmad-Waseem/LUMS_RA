import os

outRasterDir =  "D:\LUMS_RA\processed_rasters"
filelist = os.listdir(outRasterDir)

f = open("Files_to_Merge.txt","w+")
for file in filelist:
    path = os.path.join(outRasterDir,file)
    f.write('"{}"\n'.format(path))
f.close()
