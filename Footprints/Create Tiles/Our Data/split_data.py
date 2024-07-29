import os
import numpy as np
import shutil

root_folder = r"D:\LUMS_RA\Data\Seg_data\DHA_Ph12\all\images"
train_folder = r"D:\LUMS_RA\Data\Seg_data\DHA_Ph12\train"
test_folder = r"D:\LUMS_RA\Data\Seg_data\DHA_Ph12\test"

if not os.path.exists(os.path.join(train_folder, 'images')):
    os.makedirs(os.path.join(train_folder, 'images'))
if not os.path.exists(os.path.join(train_folder, 'masks')):
    os.makedirs(os.path.join(train_folder, 'masks'))
if not os.path.exists(os.path.join(train_folder, 'georeferenced_images')):
    os.makedirs(os.path.join(train_folder, 'georeferenced_images'))

if not os.path.exists(os.path.join(test_folder, 'images')):
    os.makedirs(os.path.join(test_folder, 'images'))
if not os.path.exists(os.path.join(test_folder, 'masks')):
    os.makedirs(os.path.join(test_folder, 'masks'))
if not os.path.exists(os.path.join(test_folder, 'georeferenced_images')):
    os.makedirs(os.path.join(test_folder, 'georeferenced_images'))

images = os.listdir(root_folder)
images = np.random.permutation(images)
split = 0.8

num_train = int(len(images) * split)
print(num_train)

train_ids = images[:num_train]
for id in train_ids:
    img_src = os.path.join(root_folder, id)
    msk_src = os.path.join(root_folder.replace('images', 'masks'), "mask_{}".format(id)).replace('.png', '.geojson')
    geo_src = os.path.join(root_folder.replace('images', 'georeferenced_images'), id).replace('.png', '.tif')

    img_dst = os.path.join(train_folder, 'images', id)
    msk_dst = os.path.join(train_folder, 'masks', "mask_{}".format(id)).replace('.png', '.geojson')
    geo_dst = os.path.join(train_folder, 'georeferenced_images', id).replace('.png', '.tif')

    shutil.copy(img_src, img_dst)
    try:
        shutil.copy(msk_src, msk_dst)
    except:
        print("Mask does not exist for {}".format(id))
    shutil.copy(geo_src, geo_dst)

test_ids = images[num_train:]
for id in test_ids:
    img_src = os.path.join(root_folder, id)
    msk_src = os.path.join(root_folder.replace('images', 'masks'), "mask_{}".format(id)).replace('.png', '.geojson')
    geo_src = os.path.join(root_folder.replace('images', 'georeferenced_images'), id).replace('.png', '.tif')

    img_dst = os.path.join(test_folder, 'images', id)
    msk_dst = os.path.join(test_folder, 'masks', "mask_{}".format(id)).replace('.png', '.geojson')
    geo_dst = os.path.join(test_folder, 'georeferenced_images', id).replace('.png', '.tif')

    shutil.copy(img_src, img_dst)
    try:
        shutil.copy(msk_src, msk_dst)
    except:
        print("Mask does not exist for {}".format(id))
    shutil.copy(geo_src, geo_dst)