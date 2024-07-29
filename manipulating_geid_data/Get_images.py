import cv2
import glob
import shutil
import os
from tqdm import tqdm

input_directory = "D:\LUMS_RA\Google_Earth_Images_Downloader\Complete_Zameen"
dirs = [dir for dir in os.listdir(input_directory) if not os.path.isfile(dir)]

for dir in dirs:
    if dir == "17":
        #print(dir)
        inp_path = os.path.join(input_directory,dir)+"\**\**\**\*.jpg"
        images = [file for file in glob.glob(inp_path)]
        
        out_folder = os.path.join(input_directory,dir)+"\images"
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        print(len(images))
        #break

        for image in tqdm(images):
            #print(image)
            image_name = image.split('\\')[-1]
            destination_path = out_folder+"\\"+image_name
            #print(destination_path)
            #break
            shutil.move(image, destination_path)
            print("")
            #break
        #break

