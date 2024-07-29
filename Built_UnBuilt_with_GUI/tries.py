import numpy as np
from rasterio.features import shapes
import skimage
from shapely.geometry import shape

img = np.load(r"D:\Output\clipped_preds.npy")
labels = skimage.measure.label(img, connectivity=2, background=0).astype('uint16')
print(labels.shape)
polygon_gen = shapes(labels, labels > 0)
geoms_np = []
ct = 0
for polygon, value in polygon_gen:
    ct = ct + 1
    p = (shape(polygon))
    print(type(p))
    if p.area >= 0:
        simp = p.simplify(tolerance=0.5)
        print("here")
        try:
            p = np.array(p.boundary.xy, dtype='int32').T
        except:
            p = np.array(p.boundary[0].xy, dtype='int32').T
        geoms_np.append(p)