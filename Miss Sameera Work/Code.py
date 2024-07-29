import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal import welch, detrend, filtfilt

import mne


label = ['time','TP9', 'AF7', 'AF8', 'TP10'] # channel names
sfreq = 256 # sampling frequency



eeg_time_series_table = pd.read_csv('EEG_recording_2022-11-23-09.21.08.csv',)


    # label = desc.cget("text").split(' ') # getting label values
    # sfreq = int(nominalSrate.cget("text")) # getting nominal srate value
# %matplotlib qt

info = mne.create_info(label, sfreq, ch_types='eeg') # creating info for RawArray function
AF7 = eeg_time_series_table['AF7']
AF8 = eeg_time_series_table['AF8']

lowcut = 0.5
highcut = 40.0
nyq = 0.5 * 256
low = lowcut/nyq
high = highcut/nyq
b, a = scipy.signal.butter(2, [low, high], 'bandpass', analog=False)
AF7_filt = scipy.signal.filtfilt(b, a, AF7, axis=0)
AF8_filt = scipy.signal.filtfilt(b, a, AF8, axis=0)

data_af7_detrend = detrend(AF7_filt)
data_af8_detrend = detrend(AF8_filt)
            
time = np.array(eeg_time_series_table['timestamps'])
total_time = (time[-1] - time[0])
print(total_time)
sample_rate = int((data_af7_detrend.shape / total_time).round())
print(sample_rate)
right_smple_1_AF7 = np.transpose(np.array(data_af7_detrend[sample_rate*0:sample_rate*115]).reshape((-1,115)),(1,0))
right_smple_1_AF8 = np.transpose(np.array(data_af8_detrend[sample_rate*0:sample_rate*115]).reshape((-1,115)),(1,0))
print(right_smple_1_AF7.shape)

trues = 0
total = 0
for i in range(right_smple_1_AF7.shape[0]):
    # plt.figure()
    # plt.plot(right_smple_1_AF7[i])
    # plt.figure()
    # plt.plot(right_smple_1_AF8[i])
    # # plt.legend(['AF7', 'AF8'])
    # plt.show()
    c1 = (np.count_nonzero(right_smple_1_AF7[i] > 500))
    c2 = (np.count_nonzero(right_smple_1_AF8[i] > 500))

    Actual = 'right' if (0<= i and i<30) or (60<= i and i<90) else 'left'
    Predicted = 'right' if c1 > c2 else 'left'
    print("Actual: {}, Predicted: {}, C1: {}, C2: {}".format(Actual, Predicted, c1, c2))
    if Actual == Predicted:
        trues = trues+1
    total=total+1

print(trues)
print(total)
# means = np.mean(right_smple_1_AF7, axis=-1)
# print(means.shape)
# plt.figure()
# plt.plot(means)
# means = np.mean(right_smple_1_AF8, axis=-1)
# plt.plot(means)
# plt.legend(['AF7', 'AF8'])
# # plt.show()

# left_smple_1_AF7 = np.transpose(np.array(data_af7_detrend[sample_rate*90:sample_rate*115]).reshape((-1,25)),(1,0))
# left_smple_1_AF8 = np.transpose(np.array(data_af8_detrend[sample_rate*90:sample_rate*115]).reshape((-1,25)),(1,0))
# print(right_smple_1_AF7.shape)

# means = np.mean(left_smple_1_AF7, axis=-1)
# print(means.shape)
# plt.figure()
# plt.plot(means)
# means = np.mean(left_smple_1_AF8, axis=-1)
# plt.plot(means)
# plt.legend(['AF7', 'AF8'])
# plt.show()
# right_smple_2_AF7 = np.transpose(np.array(AF7[sample_rate*60:sample_rate*90]).reshape((-1,30)),(1,0))
# right_smple_2_AF8 = np.transpose(np.array(AF8[sample_rate*60:sample_rate*90]).reshape((-1,30)),(1,0))
# print(right_smple_1_AF7.shape)

# means = np.mean(right_smple_2_AF7, axis=-1)
# print(means.shape)
# plt.plot(means)
# plt.show()
# data = np.transpose(eeg_time_series_table) # transposing the dataframe to 4(channels) x ...(number of seconds)
# ax1 = mne.io.RawArray(data, info) 

# try:
#   ax1.plot(scalings=dict(eeg=2000), title='Signal Display', bgcolor='white', color={'eeg':'black'}) # plotting the channels
# except ValueError:
#   pass




















# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

# x_axis_plt = np.arange(0,257)
# headers = ['TP9', 'AF7', 'AF8', 'TP10']

# df = pd.read_csv('draft4_data.csv', names=headers)
# tp9_col = df['TP9']

# AF7_col = df['AF7']

# AF8_col= df['AF8']

# tp10_col = df['TP10']

# print(tp10_col)

# plt.plot(x_axis_plt, tp10_col)
# plt.show()
# print
# #f.set_index(x_axis_plt).plot()
# #f.plot(x = x_axis_plt, y=headers, kind="line", figsize=(10, 10))
 
# #lt.show()