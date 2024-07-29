import pandas as pd
import glob
import os
import numpy as np


base = "New_Files/*.csv"
files = [os.path.join(os.getcwd(), path) for path in glob.glob(base)]
# print(files)

left_file = os.path.join(os.getcwd(), "Data/Left.csv")
right_file = os.path.join(os.getcwd(), "Data/Right.csv")
blink_file = os.path.join(os.getcwd(), "Data/Blink.csv")

for file in files:
    data = pd.read_csv(file)
    time = np.array(data['timestamps'])
    total_time = (time[-1] - time[0])
    total_values = time.shape[0]
    # sample_rate = int((total_values / total_time).round())
    sample_rate = 256
    extras = int(total_values % sample_rate)

    print("File: {}".format(file))
    print(total_time)
    print(total_values)
    print(sample_rate)
    print(extras)

    print(os.path.exists(left_file))
    print(os.path.exists(right_file))
    print(os.path.exists(blink_file))

    TP9 = np.array(data['TP9'][extras:]).reshape(sample_rate, -1).T
    AF7 = np.array(data['AF7'][extras:]).reshape(sample_rate, -1).T
    AF8 = np.array(data['AF8'][extras:]).reshape(sample_rate, -1).T
    TP10 = np.array(data['TP10'][extras:]).reshape(sample_rate, -1).T

    left_samples = {'TP9': TP9[:5][:].reshape(-1),
                    'AF7': AF7[:5][:].reshape(-1),
                    'AF8': AF8[:5][:].reshape(-1),
                    'TP10': TP10[:5][:].reshape(-1)}
    if not os.path.exists(left_file):
        pd.DataFrame.from_dict(left_samples).to_csv(left_file, mode='w', index=False)
    else:
        pd.DataFrame.from_dict(left_samples).to_csv(left_file, mode='a', index=False, header=False)


    right_samples = {'TP9': TP9[5:10][:].reshape(-1),
                    'AF7': AF7[5:10][:].reshape(-1),
                    'AF8': AF8[5:10][:].reshape(-1),
                    'TP10': TP10[5:10][:].reshape(-1)}
    if not os.path.exists(right_file):
        pd.DataFrame.from_dict(right_samples).to_csv(right_file, mode='w', index=False)
    else:
        pd.DataFrame.from_dict(right_samples).to_csv(right_file, mode='a', index=False, header=False)

    blink_samples = {'TP9': TP9[10:][:].reshape(-1),
                    'AF7': AF7[10:][:].reshape(-1),
                    'AF8': AF8[10:][:].reshape(-1),
                    'TP10': TP10[10:][:].reshape(-1)}
    if not os.path.exists(blink_file):
        pd.DataFrame.from_dict(blink_samples).to_csv(blink_file, mode='w', index=False)
        print("Creating File")
    else:
        pd.DataFrame.from_dict(blink_samples).to_csv(blink_file, mode='a', index=False, header=False)

    # print(len(left_samples['TP9']))
    # print(len(right_samples['TP9']))
    # print(len(blink_samples['TP9']))

