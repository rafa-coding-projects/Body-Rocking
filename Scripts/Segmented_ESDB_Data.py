import numpy as np
import scipy.io as sio
import os
import platform
from scipy.signal import butter, lfilter


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data, axis=1)
    return y


def concatenate_sessions(var, pair, sensor):


    first_time = True
    for session in pair:

        if first_time:
            cur_data = var[session][sensor]
            cur_label = var[session]['labels'].ravel()
            first_time = False
        else:
            cur_data = np.concatenate((var[session][sensor], cur_data), axis=0)
            cur_label = np.concatenate((var[session]['labels'].ravel(), cur_label), axis=0)

    return cur_data, cur_label


def load_session_ESDB(dataset_path, datafile):


    vars = []
    for (dirname, dirs, files) in os.walk(os.path.join(os.path.expanduser('~'), dataset_path)):
        for filename in files:
            if filename.startswith(datafile):
                # print('found! {}'.format(datafile))
                vars.append(sio.loadmat(os.path.join(dirname, filename)))

    sub1_data, sub1_labels = concatenate_sessions(vars, [0, 1], 'gyro_mw')
    sub2_data, sub2_labels = concatenate_sessions(vars, [2, 3], 'gyro_mw')
    sub3_data, sub3_labels = concatenate_sessions(vars, [4, 5], 'gyro_mw')
    sub4_data, sub4_labels = concatenate_sessions(vars, [6, 7], 'gyro_mw')
    sub5_data, sub5_labels = concatenate_sessions(vars, [8, 9], 'gyro_mw')
    sub6_data, sub6_labels = concatenate_sessions(vars, [10, 11], 'gyro_mw')
    sub7_data, sub7_labels = concatenate_sessions(vars, [12, 13], 'gyro_mw')

    data = [sub1_data, sub2_data, sub3_data, sub4_data, sub5_data, sub6_data, sub7_data]
    labels = [sub1_labels, sub2_labels, sub3_labels, sub4_labels, sub5_labels, sub6_labels, sub7_labels]

    return data, labels

# ---------------------------------------------------------------------------------------
#         Aux. Functions
# ---------------------------------------------------------------------------------------
if platform.system() != 'Windows':
    orig_path = r'RockingMotion/Journal/ESDB_dataset'
    load_model_path = r'RockingMotion/Journal/src'
else:
    load_model_path = r'C:\Users\rdasilv2\Gdrive\Backup Rafael\Documents\NC State\Research Related\ML and DSP\Proj - Body Rocking\src\transferlearning_ESDB_equipped_3'
    orig_path = r'C:\Users\rdasilv2\Gdrive\Backup Rafael\Documents\NC State\Research Related\ML and DSP\Proj - Body Rocking\ESDB_dataset'

data_file_name = 'Arm_100_resamp.mat'
data, labels = load_session_ESDB(orig_path, data_file_name)

samplingFreq = 100
overlap = 10

FSAMP = 100
Low = 0.1
High = 45
FONDOSCALA_ACC = 16
FONDOSCALA_GYR = 2000

accCoeff = FONDOSCALA_ACC*4/32768.0
gyrCoeff = FONDOSCALA_GYR/32768.0
magCoeff = 0.007629


for i in range(len(data)):

    data[i] = butter_bandpass_filter(data[i] * gyrCoeff, Low, High, FSAMP)
    sampleNum = data[i].shape[0]//samplingFreq*samplingFreq//overlap-10

    channelNum = data[i].shape[1]
    X = np.zeros([sampleNum, samplingFreq, channelNum])

    Y = np.zeros([sampleNum, ], dtype=int)

    for s in range(sampleNum):
        X[s, :, :] = data[i][s*overlap:s*overlap+samplingFreq,:]
        if np.mean(labels[i][s*overlap:s*overlap+samplingFreq]) > 0.5:
            Y[s, ] = 1

    # ----save prepared features for CNN architecture.
    sio.savemat(os.path.join(orig_path, 'subject' + str(i) + '.mat'), {'X': X, 'Labels': Y})
