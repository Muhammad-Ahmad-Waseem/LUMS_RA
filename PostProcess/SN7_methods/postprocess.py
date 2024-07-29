import os
import re
import time
import random
import sys
import multiprocessing
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import geojson
from PIL import Image
from PIL.TiffTags import TAGS
import matplotlib.pyplot as plt
import cv2
import skimage.io
from skimage.draw import polygon
from skimage import measure
from skimage import color as clr
from skimage.segmentation import watershed
from skimage.segmentation import expand_labels
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

#from rasterio import features
import rasterio

import solaris as sol
from shapely.ops import cascaded_union
from shapely.geometry import shape, Polygon

# y axis is flipped / mirrored
y_flip = True


def save_as_csv(out_path, contours):
    out_path = os.path.join(out_path, 'polygons.csv')
    print('save csv: %s, npoly = %d' % (out_path, len(contours)))
    fw = open(out_path, 'w')
    fw.write("filename,id,Geometry\n")
    for j, contour in enumerate(contours):
        polygon_str = re.sub(r"[\[\]]", '', ",".join(map(str, contour)))
        #polygon_str = polygon_str.replace(". ", ' ')
        #polygon_str = polygon_str.replace(".,", ',')
        #polygon_str = re.sub(r" {2,}", ' ', polygon_str)
        #polygon_str = re.sub(r" {0,}, {0,}", ',', polygon_str)

        # print(polygon_str)
        #points = polygon_str.split(",")
        #poly_str = ""
        #polygon_str = poly_str[:-1]
        fw.write("%s,%d,\"POLYGON ((%s))\"\n" % ("2022_09_12", j, polygon_str))
    fw.close()

'''
This function takes geojson point-file along with its corresponding overlay image
and returns an image where corresponding pixels are one.
'''
def pixelize(geo_file,img_path):
    with open(geo_file) as f:
        gj = geojson.load(f)

    geo_coords = list(geojson.utils.coords(gj['features']))
    geo_x, geo_y = zip(*geo_coords)

    dataset = rasterio.open(img_path)
    x_min_coord = dataset.bounds[0] #left edge
    x_max_coord = dataset.bounds[2] #right edge
    y_min_coord = dataset.bounds[1] #bottom edge
    y_max_coord = dataset.bounds[3] #top edge
    y_img_size, x_img_size = dataset.read(1).shape
    dataset.close()

    x = (((np.array(geo_x) - x_min_coord)/(x_max_coord - x_min_coord)) * (x_img_size-1)).round().astype(int)
    y = (((np.array(geo_y) - y_min_coord)/(y_max_coord - y_min_coord)) * (y_img_size-1)).round().astype(int)
    if(y_flip):
        y = y_img_size-y-1
    coords = np.vstack((x, y)).T

    plot_mask = np.zeros((x_img_size,y_img_size), dtype=bool)
    plot_mask[tuple(coords.T)] = True

    return plot_mask.T
'''
This function takes the contours made from pixels and map them to geo-coords.
The mapping is computed using cordinates of its corresponding raster.
'''
def assign_geocoords(contours,imgs_dir,img_path):
    #img_path = get_respond_img(imgs_dir,npy_file)
    dataset = rasterio.open(img_path)

    x_min_coord = dataset.bounds[0] #left edge
    x_max_coord = dataset.bounds[2] #right edge
    y_min_coord = dataset.bounds[1] #bottom edge
    y_max_coord = dataset.bounds[3] #top edge

    y_img_size,x_img_size = dataset.read(1).shape
    dataset.close()

    for it in range(len(contours)):
        contour = contours[it]
        
        x = (contour[:,0])
        y = (contour[:,1])
        if(y_flip):
            y = y_img_size - (y)
            
        geo_x = x_min_coord + (x/x_img_size)*(x_max_coord - x_min_coord)
        geo_y = y_min_coord + (y/y_img_size)*(y_max_coord - y_min_coord)

        contours[it] = np.vstack((geo_x, geo_y)).T

    return contours

'''
This function returns the path to corresponding image file of generated output
'''
def get_respond_img(imgs_dir,npy_file):
    file_name = os.path.split(npy_file)[-1].replace(".npy", ".tif")
    src_img_path = os.path.join(imgs_dir, file_name)
    return src_img_path

''' ============================ Building Detection =============================
    The function is an implementation of building detection using watershed (as desc-
    ribed by SpaceNet Winner solution). Here:

    temporal_collapse = temporaly collapsed image
    Beta_h = threshold decribed in equation
    Beta_l = threshold decribed in equation
    distance = # of pixels to consider while calculating local maxima
    min_area = minimum area for a polygon to have
    polygon_buffer = apply buffer to polygons
    conn = number of hops for measuring labels
    watershed_line = include 1 pixel gap between consective polygons or not

    Returns:
    contours = boundries of each polygon detected (a list of connected pts).
    polygons = detected polygons (a tuple containing all connecting pts).
    
    ================================= Building Detection ========================= '''
def get_building_polygon(temporal_collapse, Beta_h, Beta_l, distance, min_area, polygon_buffer, conn=2, watershed_line=True, plot_locs=False, imgs=None,
                         geofiles=None, out=None):

    plt.imsave(os.path.join(out, "1_Input_Image.png"), temporal_collapse)
    
    print("Starting Extracting Polygons..!")

    print("Calculating start locations...")
    '''
    for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]:
        mask = temporal_collapse > i
        #M1 = peak_local_max(temporal_collapse, indices=False, min_distance=distance,labels=(temporal_collapse > Beta_l))
        M1 = mask
        plt.imsave('D:\\LUMS_RA\\Python Codes\\PostProcess\\SN7_methods\\test\\M1_{}.png'.format(i),M1)
    '''
    locs = np.load(r"D:\LUMS_RA\Data\Google_Earth_Images_Downloader\DHA\19\12_09_2022.npy")
    if (plot_locs):
        img = os.path.join(imgs,os.listdir(imgs)[0])
        AOI = '_'.join(os.path.split(img)[-1].split('.')[0].split('_')[:-1])
        geofile = os.path.join(geofiles,AOI+".geojson")
        start_locs = pixelize(geofile,img)
        #start_locs = ndi.binary_dilation(pixels,np.ones((7,7)))
        print(np.count_nonzero(start_locs))
        mask = (temporal_collapse>0)&(start_locs)
        start_locs[~mask] = False
        start_locs = start_locs.astype(temporal_collapse.dtype)
        print(np.count_nonzero(start_locs))
        #dst = cv2.addWeighted(temporal_collapse,1,start_locs,2,0)
        #plt.imshow(dst)
        #plt.show()
        start_pts, num_feats = ndi.label(start_locs)
        start_pts = expand_labels(start_pts, 7)
        print(num_feats)
        #return 0,0
    else:
        M1 = locs >= Beta_h
        #M1 = peak_local_max(temporal_collapse, indices=False, footprint=np.ones((distance*2+1, distance*2+1)),labels=(temporal_collapse > Beta_l))
        start_locs = M1
        start_pts, num_feats = ndi.label(start_locs)

    plt.imsave(os.path.join(out, "2_Pts.png"), start_pts)
    #plt.imsave('D:\\LUMS_RA\\Python Codes\\PostProcess\\SN7_methods\\test\\Pts.png',start_pts)
    print("Completed.")
    #return 0,0

    print("Running watershed...")
    ws_out = watershed(image=(-temporal_collapse), markers=start_pts, mask=(temporal_collapse > 0), watershed_line=watershed_line)
    plt.imsave(os.path.join(out, "3_WS.png"), ws_out)
    #plt.imsave('D:\\LUMS_RA\\Python Codes\\PostProcess\\SN7_methods\\test\\WS.png',ws_out)
    #im2 = ((ws_out+start_pts)>0).astype(temporal_collapse.dtype)
    #plt.imshow(temporal_collapse+im2)
    #plt.show()
    print("Watershed completed.")
    #return 0,0
    
    print("Running measure labels...")
    labels = measure.label(ws_out, connectivity=conn, background=0).astype('uint16')
    #plt.imsave(os.path.join(out, "label.png"), labels)
    np.save(os.path.join(out, "4_label.png"), labels)
    #plt.imsave('D:\\LUMS_RA\\Python Codes\\PostProcess\\SN7_methods\\test\\label.png',labels)
    print("Measure labels completed.")
    #return 0,0
    

    geoms_np = []
    geoms_polygons = []
    polygon_generator = rasterio.features.shapes(labels, labels>0)

    for polygon, value in polygon_generator:
        p = shape(polygon)
        if polygon_buffer:
            p = p.buffer(polygon_buffer)
        if p.area >= min_area:
            p = p.simplify(tolerance=0.5)
            geoms_polygons.append(p)
            try:
                p = np.array(p.boundary.xy, dtype='int32').T
            except:
                p = np.array(p.boundary[0].xy, dtype='int32').T
                
            geoms_np.append(p)
    
    return geoms_np, geoms_polygons

'''
This function will be used to get a list of pixels inside a contour
'''
def get_idx_in_polygon(contour):
    contour = contour.reshape(-1, 2) - 0.5
    p = Polygon(contour).buffer(-0.25)
    r, c = p.boundary.xy
    rr, cc = polygon(r, c)
    return rr, cc

'''
The function which processes outputs of each AOI, called iteratively by main()
'''
def process(npy_list=None, thres1=0.5, thres2_h1=0.9999, thres2_l1=0.7, thres2_h2=0.6, thres2_l2=0.35,
            thres3_1=0.3, thres3_s=0.45, thres3_d=0.5, thres3_i=0, thres3_m=0.3,
            margin=0, distance = 5, min_area=25.5, polygon_buffer=0):

    root = sys.argv[1]
    #imgs_dir = os.path.join(root,"Images")
    imgs_dir = root
    geofiles_dir = os.path.join(root, "Locations")
    out_dir = os.path.join(root, "Outputs")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    '''
    #npy_list = sorted(npy_list)

    #To sum up all temporal predictions for that aoi
    npy_sum = 0

    #To take into account pixels > alpha (thres1)
    ignore_sum = 0

    #To create temporal array of (h x w x t) for that aoi
    npys = []
    
    for iii, npy_file in enumerate(npy_list):
        npy = np.load(npy_file)
        
        # uncomment this part, if you want to ignore zero pixels of raster image
        img_file = get_respond_img(imgs_dir,npy_file)
        src_img = skimage.io.imread(img_file)
        print(src_img.shape)
        mask = src_img[:,:,3]==0
        mask = np.repeat(mask, 3, axis=0)
        mask = np.repeat(mask, 3, axis=1)
        assert mask.shape[0]== src_img.shape[0] * 3 and mask.shape[1] == src_img.shape[1] * 3
        npy = npy[:mask.shape[0], :mask.shape[1]]
        npy[mask] = -10000
        npys.append(npy)
        #ignore_mask = (npy >= thres1)
        #npy_sum = npy_sum + npy * ignore_mask
        #ignore_sum = ignore_sum + ignore_mask
        break
        
        #print("Npy File : {}, Non Zero in mask: {}".format(npy_file,np.count_nonzero(ignore_sum)))
    #plt.imsave('D:\\LUMS_RA\\Python Codes\\PostProcess\\SN7_methods\\test\\NS.png',npy_sum)
    #return
    #print(np.count_nonzero(npy_sum))
    #npy_sum[npy_sum<2] = 0
    #xcprint(np.count_nonzero(npy_sum))
    #return
    #npy_mean = (npy_sum) / (np.maximum(ignore_sum, 1))
    #npy_mean = (npy_sum) / npys.shape[0]
    #print(npy_sum.dtype)
    #print(npy_mean.dtype)
    #print(np.unique(npy_mean))
    #return
    #assert len(npy_list) > 1, "Total npy lists are more than 1, Select a single image"
    #npys = np.array(npy_list[0],dtype=np.float16)
    #img_num = npys.shape[0]
    '''
    contours, polygons = get_building_polygon(np.load(r"D:\LUMS_RA\Data\Google_Earth_Images_Downloader\DHA\19\12_09_2022_mask.npy"), thres2_h1, thres2_l1, distance=distance, min_area=min_area,
                                              polygon_buffer=polygon_buffer, imgs=imgs_dir, geofiles= geofiles_dir,
                                              out = out_dir)

    geo_poly = (assign_geocoords(contours, "", r"D:\LUMS_RA\Data\Google_Earth_Images_Downloader\DHA\19\12_09_2022.tif"))
    #print(geo_poly)

    #print(geo_poly.shape)

    save_as_csv(r"D:\LUMS_RA\Data\Google_Earth_Images_Downloader\DHA\19\Outputs", geo_poly)
    return
    #save_as_csv(out_dir, contours, npy_list)
    #return

    building_num = len(contours)

    score_map = np.zeros((building_num, img_num))

    footprints = np.zeros_like(score_map)

    changeids = []
    for i, contour in enumerate(contours):
        rr = (contour[:,0])
        cc = (contour[:,1])
        point_filter = (rr < npys.shape[2]) & (cc < npys.shape[1]) & (cc >= 0) & (rr >= 0)
        rr = rr[point_filter]
        cc = cc[point_filter]

        scores = np.mean(npys[:, cc, rr], axis=1)
        score_mask = scores >= 0
        max_score = scores.max()
        masked_scores = scores[score_mask]

        left_mean = np.cumsum(masked_scores)/(np.arange(len(masked_scores))+1)
        right_mean = (np.cumsum(masked_scores[::-1])/(np.arange(len(masked_scores))+1)) [::-1]

        max_diff = 0
        state_change = False
        for idx in range(len(masked_scores)-1):
            diff = right_mean[idx+1] - left_mean[idx]
            max_diff = max(diff, max_diff)
            if max_diff > thres3_d:
                state_change = True
                break

        if state_change:
            score_map[i][idx:] = np.ones((score_map[i][idx:]).shape)
        elif np.mean(masked_scores) > thres3_m:
            score_map[i][:] = np.ones((score_map[i][:]).shape)

        print("Index: {}, State Change Flag: {}, Mean Score: {}, Scores: {}".format(i,state_change,np.mean(masked_scores),scores))
        continue
        if max_diff > thres3_d:
            changeids.append(i)
            start = False
            for idx, score in enumerate(scores):
                if not start:
                    if idx == 0 and score > thres3_1:
                        score_filter[idx] = 1
                        start = True
                    if score > thres3_s * max_score:
                        score_filter[idx] = 1
                        start = True
                else:
                    if score > thres3_i:
                        score_filter[idx] = 1

        footprints[i] = score_filter

    return
    changeids = np.array(changeids)
    change_contours = [contours[idx] for idx in changeids]
    change_polygons = [polygons[idx] for idx in changeids]
    change_footprints = footprints[changeids]

    geo_poly = assign_geocoords(contours,imgs_dir,npy_list[0])
    save_as_csv(out_dir,contours,npy_list)

    return

    # print('num change footprints:', len(changeids), change_footprints.sum())

    ''' ============================ For Tracking ============================ '''
    contours, polygons = get_building_polygon(npy_mean, thres2_h2, thres2_l2,
                                              distance=distance, min_area=min_area, polygon_buffer=polygon_buffer)
    # print('num track footprints (before filter):', len(contours))
    filter_idx = filter_countours(polygons, change_polygons, margin)
    contours = [contours[idx] for idx in filter_idx]
    # print('num track footprints (after filter):', len(contours))

    building_num = len(contours)
    score_map = np.zeros((building_num, img_num))

    footprints = np.zeros_like(score_map)
    for i, contour in enumerate(contours):
        rr = (contour[:,0])
        cc = (contour[:,1])

        point_filter = (rr < npys.shape[2]) & (cc < npys.shape[1]) & (cc >= 0) & (rr >= 0)
        rr = rr[point_filter]
        cc = cc[point_filter]
        
        scores = np.mean(npys[:, rr, cc], axis=1)
        score_mask = scores >= 0

        score_filter = np.zeros(img_num, dtype='bool')
        max_score = scores.max()

        masked_scores = scores[score_mask]
        left_mean = np.cumsum(masked_scores)/(np.arange(len(masked_scores))+1)
        right_mean = (np.cumsum(masked_scores[::-1])/(np.arange(len(masked_scores))+1))[::-1]
        
        if scores[scores>=0].mean() > thres3_m:
            score_filter[scores>=0] = 1

        footprints[i] = score_filter

    final_contours = change_contours + contours
    final_footprints = np.concatenate([change_footprints, footprints], 0)
    save_as_csv(out_file, npy_list, final_contours, final_footprints, npys)

'''
main function which divides data into AOIs and calls process function for each AOI.
'''
def main():

    root = sys.argv[1]
    #pred_root = os.path.join(root,"vis_compose")
    '''
    dic = {}
    npy_files = [os.path.join(root, x) for x in os.listdir(root) if x.endswith('.npy')]
    for npy_file in npy_files:
        key = '_'.join(os.path.split(npy_file)[-1].split('.')[0].split('_')[:-1])
        if key not in dic:
            dic[key] = [npy_file]
        else:
            dic[key].append(npy_file)
    
    
    params = []
    
    for aoi, npy_list in dic.items():
        print("Process: {}, #files: {}".format(aoi,len(npy_list)))
        params.append(npy_list)  
    '''

    print("Execute!")
    #print("len params:", len(params))
    n_threads = 1
    #pool = multiprocessing.Pool(n_threads)
    #_ = pool.map(process, params)

    _ = process()
    print("Finish!")


if __name__ == "__main__":
    main()
