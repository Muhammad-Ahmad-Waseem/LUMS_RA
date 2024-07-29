import sys

import data_lib as dlib

# I am assuming that path contains path to directory where all tiff files
# are placed in a folder named "Images" (case sensitive)

path = sys.argv[1]
mode = sys.argv[2]

if mode == "train":
    dlib.create_label(path, f3x=False)
    dlib.enlarge_3x(path)
    dlib.create_label(path, f3x=True)
    dlib.divide(path)
    dlib.create_trainval_list(path)
elif mode == "test":
    dlib.divide(path)
    dlib.create_test_list(path)
elif mode == "compose":
    dlib.compose(path)
else :
    dlib.compose(path, ext=".png")
