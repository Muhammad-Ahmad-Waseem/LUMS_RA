# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 15:33:19 2022

@author: Sohaib Bin Mohsin

1- RUN animation for training subjects to play pong using 3 eye movments: eye-blinks and horizontal
Lt-Rt eye-movements.
1a-stream in EEG data when anim starts and save in .csv


2-preprocess csv files:

3-prepare fgive labels 
mne-bandpass filter removed

scipy-butter filter used instead


"""



'''
Name: 
Input: 
Process: 
Returns: None
'''   

from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
import numpy as np
import pandas as pd
from time import sleep, perf_counter
from scipy.signal import butter, lfilter, lfilter_zi
import subprocess
import sys
from threading import Thread
from pynput.keyboard import Key, Controller
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

keyboard_game = Controller()

global inlet, fs, eeg_buffer

NOTCH_B, NOTCH_A = butter(4, np.array([55, 65]) / (256 / 2), btype='bandstop')
# from:https://github.com/alexandrebarachant/muse-lsl/commit/eb04fceab0d7d2ce84494124149a7e804bca335a#diff-c5a63aec088a3950aa3c8324ae7115dc1f93b11ee571e4f32bafb0894fd85ca3

'''
-----------------------------------------------------------------------------------
Name: playGame()
Input: python game file
Process: plays the pong game file using os system call
Returns: None
'''    
def playGame():
    subprocess.call([sys.executable, 'pong-single (2).py', ])


''''
 ----------------------------------------------------------------------------------------------------
Name: update_buffer()
Input: data_buffer, new_data, label, notch=False, filter_state=None
Process: 
    Concatenates "new_data" into "data_buffer", and returns an array with
    the same size as "data_buffer"
    
Returns:new_buffer, filter_state

'''  
def update_buffer(data_buffer, new_data, label, notch=False, filter_state=None):
    
    if new_data.ndim == 1:
        new_data = new_data.reshape(-1, data_buffer.shape[1])

    if notch:
        if filter_state is None:
            filter_state = np.tile(lfilter_zi(NOTCH_B, NOTCH_A),
                                   (4, 1)).T
        new_data, filter_state = lfilter(NOTCH_B, NOTCH_A, new_data, axis=0,
                                         zi=filter_state)
    
    label_data = []
    if label == 1:
        label_data = np.ones((new_data.shape[0], 1), dtype=int)
    else:
        label_data =  np.zeros((new_data.shape[0], 1), dtype=int)
   
    label_data = np.append(new_data, label_data, axis=1)
    new_buffer = np.concatenate((data_buffer, label_data), axis=0)
    new_buffer = new_buffer[new_data.shape[0]:, :]

    return new_buffer, filter_state

'''
 ----------------------------------------------------------------------------------------------------
Name: get_data(label)
Input: label
Process: Obtain EEG data from the LSL stream and give it appropriate label acc to the blink,rt+lt moves
Returns: None
'''     
def get_data(label):
    global eeg_buffer
    filter_state = None
    # Obtain EEG data from the LSL stream
    info = inlet.info()
    fs = int(info.nominal_srate())
    while(fs <= 250 and fs >= 260):
        print("Sampling rate not between 250 and 260. It is: ", fs)#needs checking
        info = inlet.info()
        fs = int(info.nominal_srate())
    eeg_data, timestamp = inlet.pull_chunk(
        timeout=1, max_samples=int(SHIFT_LENGTH * fs))
    # Index of the channel(s) (electrodes) to be used
    # 0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear
    INDEX_CHANNEL = [0, 1, 2, 3]
    # Only keep the channel we're interested in
    ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]
    # print(ch_data)
    # Update EEG buffer with the new data
    eeg_buffer, filter_state = update_buffer(
        eeg_buffer, ch_data, label, notch=True, filter_state=filter_state)

'''
 ----------------------------------------------------------------------------------------------------
Name: move()
Input:key, times, label
Process: helper for begin() 
Returns: None
'''     
def move(key, times, label):
    for i in range(times):
        get_data(label)
        sleep(1)
        keyboard_game.press(key)
        sleep(0.3)
        keyboard_game.release(key)
        sleep(0.5)

'''
 ----------------------------------------------------------------------------------------------------
Name: begin()
Input: none
Process: formatting and saving pong-animation eeg data in 3 separate files
Returns: None
'''     
def begin():
    keyr = Key.right  # gaze right
    keyl = Key.left  # looking left
    keyu = Key.up  # for blinks
    sleep(4)

    for i in range(3):
        labelBlink = 1
        labelLeft = 0
        labelRight = 0
        if(i == 0):
            labelBlink = 1
            labelLeft = 0
            labelRight = 0
            move(keyu, 12, labelBlink)
            move(keyl, 3, labelLeft)
            move(keyr, 6, labelRight)
            move(keyl, 3, labelLeft)
            global eeg_data1
            eeg_data1 = pd.DataFrame(data=eeg_buffer, columns=['TP9', 'AF7', 'AF8', 'TP10', 'Label'])
            eeg_data1.to_csv('initial-data-new-6-'+str(i+1)+'.csv', index=False)
            print("Blink buffer saved to file")
            
        elif i == 1:
            labelBlink = 0
            labelLeft = 1
            labelRight = 0
            move(keyl, 3, labelLeft)
            move(keyr, 6, labelRight)
            move(keyl, 3, labelLeft)
            move(keyr, 3, labelRight)
            move(keyl, 6, labelLeft)
            move(keyr, 3, labelRight)
            global eeg_data2
            eeg_data2 = pd.DataFrame(data=eeg_buffer, columns=['TP9', 'AF7', 'AF8', 'TP10', 'Label'])
            eeg_data2.to_csv('initial-data-new-6-'+str(i+1)+'.csv', index=False)
            print("Left buffer saved to file")
            
        else:
            labelBlink = 0
            labelLeft = 0
            labelRight = 1
            move(keyl, 3, labelLeft)
            move(keyr, 6, labelRight)
            move(keyl, 3, labelLeft)
            move(keyr, 3, labelRight)
            move(keyl, 6, labelLeft)
            move(keyr, 3, labelRight)
            global eeg_data3
            eeg_data3 = pd.DataFrame(data=eeg_buffer, columns=['TP9', 'AF7', 'AF8', 'TP10', 'Label'])
            eeg_data3.to_csv('initial-data-new-6-'+str(i+1)+'.csv', index=False)
            print("Right buffer saved to file")
			
            
'''
 ----------------------------------------------------------------------------------------------------
Name: classification()
Input: none
Process: uses 3 csv files made in begin() to train 3 classifiers and develop 3 classifier-models to 
predict in real-time for 3 moves
Returns: None
'''             
def classification():
    # keyboard_game.press('a')
    # keyboard_game.release('a')
    global eeg_data1, eeg_data2, eeg_data3
    
    # COMMENT LINES 157-159 WHILE RECORDING NEW DATA AND IF USING THE PREVIOUS DATA UNCOMMENT IT
    # eeg_data1 = pd.read_csv('initial-data-new-6-1.csv')
    # eeg_data2 = pd.read_csv('initial-data-new-6-2.csv')
    # eeg_data3 = pd.read_csv('initial-data-new-6-3.csv')
    
    # Dropping rows with NaN values
    eeg_data1 = eeg_data1.dropna(axis=0).reset_index(drop=True)
    eeg_data2 = eeg_data2.dropna(axis=0).reset_index(drop=True)
    eeg_data3 = eeg_data3.dropna(axis=0).reset_index(drop=True)
    
    # Dropping rows with all zeros
    eeg_data1 = eeg_data1.loc[~(eeg_data1==0).all(axis=1)]
    eeg_data2 = eeg_data2.loc[~(eeg_data2==0).all(axis=1)]
    eeg_data3 = eeg_data3.loc[~(eeg_data3==0).all(axis=1)]
    
    # Dividing data into Xs and ys
    eeg_data1['Mean'] = eeg_data1.mean(axis=1)
    X1 = eeg_data1[['Mean']]
    y1 = eeg_data1['Label']
    
    X2 = eeg_data2[['AF7', 'AF8']]
    y2 = eeg_data2['Label']
    
    X3 = eeg_data3[['AF7', 'AF8']]
    y3 = eeg_data3['Label']
    
    # Spliting train and test dataset
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1 , test_size=0.25, random_state=0)
    X_train2, X_test2, y_train2, y_test2= train_test_split(X2, y2 , test_size=0.25, random_state=0)
    X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3 , test_size=0.25, random_state=0)
    
    # Normalize the train data for numerical stability
    ss_train = StandardScaler()
    X_train1 = ss_train.fit_transform(X_train1)
    X_train2 = ss_train.fit_transform(X_train2)
    X_train3 = ss_train.fit_transform(X_train3)
    
    # Normalize the test data for numerical stability
    ss_test = StandardScaler()
    X_test1 = ss_test.fit_transform(X_test1)
    X_test2 = ss_test.fit_transform(X_test2)
    X_test3 = ss_test.fit_transform(X_test3)
    
    # Predictions
    global model1
    model1 = RandomForestClassifier().fit(X_train1, y_train1)
    predictions1 = model1.predict(X_test1)
    
    accuracy1 = accuracy_score(predictions1, y_test1)
    precision1 = precision_score(predictions1, y_test1)
    recall1 = recall_score(predictions1, y_test1)
    print("Accuracy of blink classifier: ", accuracy1)
    print("Precision of blink classifier: ", precision1)
    print("Recall of blink classifier: ", recall1)
    
    global model2
    model2 = RandomForestClassifier().fit(X_train2, y_train2)
    predictions2 = model2.predict(X_test2)
    
    accuracy2 = accuracy_score(predictions2, y_test2)
    precision2 = precision_score(predictions2, y_test2)
    recall2 = recall_score(predictions2, y_test2)
    print("Accuracy of blink classifier: ", accuracy2)
    print("Precision of blink classifier: ", precision2)
    print("Recall of blink classifier: ", recall2)
    
    global model3
    model3 = RandomForestClassifier().fit(X_train3, y_train3)
    predictions3 = model3.predict(X_test3)
    
    accuracy3 = accuracy_score(predictions3, y_test3)
    precision3 = precision_score(predictions3, y_test3)
    recall3 = recall_score(predictions3, y_test3)
    print("Accuracy of blink classifier: ", accuracy3)
    print("Precision of blink classifier: ", precision3)
    print("Recall of blink classifier: ", recall3)
    print("")
    print("Classification completed!")
    # keyboard_game.press('a')
    # keyboard_game.release('a')
    keyboard_game.press('b')
    keyboard_game.release('b')
    sleep(1)
    keyboard_game.press('b')
    keyboard_game.release('b')


#########################################################################################

""" EXPERIMENTAL PARAMETERS """
# Modify these to change aspects of the signal processing

# Length of the EEG data buffer (in seconds)
# This buffer will hold last n seconds of data and be used for calculations
BUFFER_LENGTH = 24

# Length of the epochs used to compute the FFT (in seconds)
EPOCH_LENGTH = 1

# Amount of overlap between two consecutive epochs (in seconds)
OVERLAP_LENGTH = 0

# Amount to 'shift' the start of each next consecutive epoch
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH

INDEX_CHANNEL = [0, 1, 2, 3]

startGame = True

if __name__ == "__main__":

    """ 1. CONNECT TO EEG STREAM """

    # Search for active LSL streams
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')

    # Set active EEG stream to inlet and apply time correction
    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    eeg_time_correction = inlet.time_correction()

    # Get the stream info and description
    info = inlet.info()
    description = info.desc()

    # Get the sampling frequency
    # This is an important value that represents how many EEG data points are
    # collected in a second. This influences our frequency band calculation.
    # for the Muse 2016, this should always be 256
    fs = 256 #int(info.nominal_srate())

    """ 2. INITIALIZE BUFFERS """

    # Initialize raw EEG data buffer
    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 5))

    """ 3. GET DATA """

    # The try/except structure allows to quit the while loop by aborting the
    # script with <Ctrl-C>
    print('Press Ctrl-C in the console to break the while loop.')

    try:
        if(startGame):
            startGame = False
            t = Thread(target=playGame, daemon=True)
            t.start()
        # COMMENT LINE 307 IF YOU ARE USING OLD DATA ELSE UNCOMMENT IT WHEN RECORDING NEW DATA
        begin()
        classification()
        keyboard_game.press('c')
        keyboard_game.release('c')
        sleep(1)
        keyboard_game.press('c')
        keyboard_game.release('c')
        keyboard_game.press('d')
        keyboard_game.release('d')
        while True:
            """ 2.1 ACQUIRE DATA """
            start_time = perf_counter()
            # Obtain EEG data from the LSL stream
            eeg_data, timestamp = inlet.pull_chunk(timeout=1, max_samples=int(SHIFT_LENGTH * fs))

            # Only keep the channel we're interested in
            ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]
            df = pd.DataFrame(data=ch_data, columns=['TP9', 'AF7', 'AF8', 'TP10'])
            df['Mean'] = df.mean(axis=1)
            
            global model1, model2, model3
            p1 = model1.predict(df[['Mean']])# predict blink
            p2 = model2.predict(df[['AF7', 'AF8']])#predict lt eye mov
            p3 = model3.predict(df[['AF7', 'AF8']])#predict rt eye-mov
            
            blink_label = np.mean(p1)
            left_label = np.mean(p2)
            right_label = np.mean(p3)
            
            print("Blink:", blink_label)
            print("Left:", left_label)
            print("Right:", right_label)
            if blink_label >= 0.5:
                pass
            if left_label >= 0.5:
                print("Inside left")
                keyboard_game.press(Key.left)
                sleep(0.3)
                keyboard_game.release(Key.left)
            if right_label >= 0.5:
                print("Inside right")
                keyboard_game.press(Key.right)
                sleep(0.3)
                keyboard_game.release(Key.right)
            end_time = perf_counter()
            if end_time-start_time < 1:
                sleep(1-(end_time-start_time))

    except KeyboardInterrupt:
        print('Closing!')
