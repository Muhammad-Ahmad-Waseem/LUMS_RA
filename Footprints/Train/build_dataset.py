import argparse
import random
import os
import numpy as np

import shutil
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default="DHA_GEID_Current/images", help= "Directory where un-splitted data is placed")

if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.input_dir), "Couldn't find the dataset at {}".format(args.input_dir)

    dirs = [dir for dir in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir,dir))]
    for dir in dirs:
        train_dir_path = os.path.join(args.input_dir,dir)

        files = os.listdir(train_dir_path)
        files.sort()
        print("Initial size: {}".format(len(files)))

        idxs = list(range(len(files)))
        random.seed(230)
        random.shuffle(idxs)

        test_idxs = idxs[:int(0.2*len(files))]
        test_files = np.array(files)[test_idxs]
        test_files_paths = [os.path.join(train_dir_path, f) for f in test_files if f.endswith('.png')]

        test_dir_path = os.path.join(args.input_dir, "test_" + dir)
        if not os.path.exists(test_dir_path):
            os.makedirs(test_dir_path)

        for filename in tqdm(test_files_paths):
            shutil.move(filename,os.path.join(test_dir_path, os.path.split(filename)[-1]))

        print("Final size: {}".format(len(os.listdir(train_dir_path))))

    print("Done building dataset")
