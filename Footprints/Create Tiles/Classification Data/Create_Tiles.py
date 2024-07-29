import rasterio

tile_size = 1024
tiff_path = r"D:\LUMS_RA\Data\Google_Earth_Images_Downloader\Tiff_Images_21_Zoom\Allama Iqbal Town.tif"
out_file = r'D:\LUMS_RA\Data\Classification Data\Allama_Iqbal_Town_Tiles.csv'

dataset = rasterio.open(tiff_path)
x_min_d = dataset.bounds[0]  # left edge
x_max_d = dataset.bounds[2]  # right edge
y_min_d = dataset.bounds[1]  # bottom edge
y_max_d = dataset.bounds[3]  # top edge

image = dataset.read()
image_x_pix = image.shape[2]
image_y_pix = image.shape[1]

x_delta = (x_max_d - x_min_d) / image_x_pix
y_delta = -1 * (y_max_d - y_min_d) / image_y_pix

x_tiles_num = image_x_pix // tile_size
y_tiles_num = image_y_pix // tile_size

fw = open(out_file, 'w')
fw.write("AOI,id,filename,Geometry\n")
fw.close()

idx = 0
for x_tile_idx in range(x_tiles_num):
    x1 = tile_size * x_tile_idx
    x2 = x1 + tile_size
    for y_tile_idx in range(y_tiles_num):
        y1 = tile_size * y_tile_idx
        y2 = y1 + tile_size

        x_min = float(x_min_d + x1 * x_delta)
        x_max = float(x_min_d + x2 * x_delta)
        y_max = float(y_max_d + y1 * y_delta)
        y_min = float(y_max_d + y2 * y_delta)

        polygon_str = "{} {},{} {},{} {},{} {},{} {}".format(x_min, y_min, x_max, y_min,
                                                             x_max, y_max, x_min, y_max, x_min, y_min)

        fa = open(out_file, 'a')
        fa.write("%s,%d,%s,\"POLYGON ((%s))\"\n" % ("Allama Iqbal Town", idx, '{}'.format(idx), polygon_str))
        fa.close()

        idx = idx + 1