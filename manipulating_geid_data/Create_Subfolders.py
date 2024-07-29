import shutil
import os
from tqdm import tqdm

input_directory = "D:\\LUMS_RA\\Google_Earth_Images_Downloader\\Complete_Zameen\\17\\predictions"
out_folder = os.path.join(input_directory,"images")
files = [dir for dir in os.listdir(input_directory)]

for file in tqdm(files):
    first_part = file.split(".")[0]
    parts = first_part.split("_")

    source_dir = os.path.join(input_directory,file)
    target_dir = out_folder + "\\{}\\{}".format(parts[-1],parts[1])
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    shutil.move(source_dir, target_dir)
    print("")

