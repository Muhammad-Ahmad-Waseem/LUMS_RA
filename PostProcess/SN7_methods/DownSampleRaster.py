from solaris.preproc.image import LoadImage, SaveImage, Resize

for i in range(16):
    file = "D:\LUMS_RA\Google_Earth_Images_Downloader\Johar_Town\\{}\\{}.tif".format(i+1,i+1)
    print("Downsample :", file)
    lo = LoadImage(file)
    img = lo.load()
    _, height, width = img.data.shape
    print(height)
    print(width)
    re = Resize(1024, 1024)
    img = re.resize(img, 1024, 1024)
    assert img.data.shape[1] == 1024
    assert img.data.shape[2] == 1024

    sa = SaveImage("D:\LUMS_RA\Google_Earth_Images_Downloader\Johar_Town\\{}\\{}_downsampled.tif".format(i+1,i+1))
    sa.transform(img)
