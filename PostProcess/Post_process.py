from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

Image.MAX_IMAGE_PIXELS = 1000000000
prev2 = "D:\LUMS_RA\Google_Earth_Images_Downloader\Complete_Zameen\\1\\1_preds.tif"
prev1 = "D:\LUMS_RA\Google_Earth_Images_Downloader\Complete_Zameen\\2\predictions\images_combined\images_zoom_18.tif"
curre = "D:\LUMS_RA\Google_Earth_Images_Downloader\Complete_Zameen\\3\predictions\images_combined\images_zoom_18.tif"
next1 = "D:\LUMS_RA\Google_Earth_Images_Downloader\Complete_Zameen\\4\predictions\images_combined\images_zoom_18.tif"
next2 = "D:\LUMS_RA\Google_Earth_Images_Downloader\Complete_Zameen\\5\predictions\images_combined\images_zoom_18.tif"
files = [prev2,prev1,curre,next1,next2]

new_file = np.zeros((39936, 37632),dtype='uint8')
for file in files:
    new_file = np.add(new_file, np.all(np.equal(np.asarray(Image.open(file)), (164, 113, 88)), axis=-1))

new_file = (new_file > 2)
print(np.count_nonzero(new_file))
#cv2.imwrite('image.png', binary_image)
#cv2.imwrite("D:\LUMS_RA\Google_Earth_Images_Downloader\Complete_Zameen\\3\predictions\images_combined\Post_images_zoom_18.tif", new_file)
