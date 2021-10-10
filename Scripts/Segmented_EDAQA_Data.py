import numpy as np
import scipy.io as sio
import os
import platform
from scipy.signal import butter, lfilter


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):

    b, a = butter(order, [0.05, 0.95], btype='band')
    y = lfilter(b, a, data, axis=1)
    return y


def generate_labels(idx_time, sz):

    label = np.zeros((sz, 1))
    for idx in idx_time:
        label[idx[0]:idx[1]] = 1

    return label


def concatenate_sessions(path, dct, sbj, sensor):


    first_time = True

    for i in dct[sbj]:

        var = sio.loadmat(os.path.join(path, sbj, i, data_file_name))
        if first_time:
            cur_data = var[sensor]
            cur_label = generate_labels(var['idx_time_' + sensor.split('_')[0]], cur_data.shape[0])
            first_time = False
        else:
            cur_data = np.concatenate((var[sensor], cur_data), axis=0)
            cur_label = np.concatenate((generate_labels(var['idx_time_' + sensor.split('_')[0]], var[sensor].shape[0]),
                                        cur_label), axis=0)

    return cur_data, cur_label


def generate_study_data(path, dct, subjs, sensor):

    sub1_data, sub1_labels = concatenate_sessions(path, dct, subjs[0], 'acc_mw')
    print('Sub1, share of classes - 0:{}, 1:{}'.format(sum(sub1_labels == np.unique(sub1_labels)[0])/sub1_labels.shape[0],
                                                       sum(sub1_labels == np.unique(sub1_labels)[1])/sub1_labels.shape[0]))
    sub2_data, sub2_labels = concatenate_sessions(path, dct, subjs[1], 'acc_mw')
    print('Sub2, share of classes - 0:{}, 1:{}'.format(sum(sub2_labels == np.unique(sub2_labels)[0])/sub2_labels.shape[0],
                                                       sum(sub2_labels == np.unique(sub2_labels)[1])/sub2_labels.shape[0]))
    sub3_data, sub3_labels = concatenate_sessions(path, dct, subjs[2], 'acc_mw')
    print('Sub3, share of classes - 0:{}, 1:{}'.format(sum(sub3_labels == np.unique(sub3_labels)[0])/sub3_labels.shape[0],
                                                       sum(sub3_labels == np.unique(sub3_labels)[1])/sub3_labels.shape[0]))
    sub4_data, sub4_labels = concatenate_sessions(path, dct, subjs[3], 'acc_mw')
    print('Sub4, share of classes - 0:{}, 1:{}'.format(sum(sub4_labels == np.unique(sub4_labels)[0])/sub4_labels.shape[0],
                                                       sum(sub4_labels == np.unique(sub4_labels)[1])/sub4_labels.shape[0]))
    sub5_data, sub5_labels = concatenate_sessions(path, dct, subjs[4], 'acc_mw')
    print('Sub5, share of classes - 0:{}, 1:{}'.format(sum(sub5_labels == np.unique(sub5_labels)[0])/sub5_labels.shape[0],
                                                       sum(sub5_labels == np.unique(sub5_labels)[1])/sub5_labels.shape[0]))
    sub6_data, sub6_labels = concatenate_sessions(path, dct, subjs[5], 'acc_mw')
    print('Sub6, share of classes - 0:{}, 1:{}'.format(sum(sub6_labels == np.unique(sub6_labels)[0])/sub6_labels.shape[0],
                                                       sum(sub6_labels == np.unique(sub6_labels)[1])/sub6_labels.shape[0]))

    data = [sub1_data, sub2_data, sub3_data, sub4_data, sub5_data, sub6_data]
    labels = [sub1_labels, sub2_labels, sub3_labels, sub4_labels, sub5_labels, sub6_labels]

    return data, labels


def load_session_EDAQA(path):

    # Load folder name for both studies
    subjs, dict1, dict2 = [], dict(), dict()
    for (dirname, dirs, files) in os.walk(path):
        for dr in dirs:
            if dr.startswith('y08'):
                if dirname[-6:] in dict1 and not dict1[dirname[-6:]] is None:
                    dict1[dirname[-6:]].append(dr)
                else:
                    dict1[dirname[-6:]] = [dr]
                    subjs.append(dirname[-6:])
            elif not dr.startswith('Sub'):
                if dirname[-6:] in dict2 and not dict1[dirname[-6:]] is None:
                    dict2[dirname[-6:]].append(dr)
                else:
                    dict2[dirname[-6:]] = [dr]

    # Retrieve data, concatenate sessions by subject and return list of data per subject
    print('Retrive data for Study 1')
    data1, labels1 = generate_study_data(path, dict1, subjs, 'acc_mw')  # Study 1
    print('Retrive data for Study 2')
    data2, labels2 = generate_study_data(path, dict2, subjs, 'acc_mw')  # Study 2

    return data1, data2, labels1, labels2


def filter_segment_and_save(data, labels, path, FSAMP, High, samplingFreq):
    for i in range(len(data)):

        data[i] = butter_bandpass_filter(data[i] * accCoeff, Low, High, FSAMP)
        sampleNum = data[i].shape[0] // samplingFreq * samplingFreq // overlap - 10

        channelNum = data[i].shape[1]
        X = np.zeros([sampleNum, samplingFreq, channelNum])

        Y = np.zeros([sampleNum, ], dtype=int)

        for s in range(sampleNum):
            X[s, :, :] = data[i][s * overlap:s * overlap + samplingFreq, :]
            if np.mean(labels[i][s * overlap:s * overlap + samplingFreq]) > 0.5:
                Y[s,] = 1

        # ----save prepared features for CNN architecture.
        sio.savemat(os.path.join(path, 'subject' + str(i) + '.mat'), {'X': X, 'Labels': Y})

# ---------------------------------------------------------------------------------------
#         Aux. Functions
# ---------------------------------------------------------------------------------------
if platform.system() != 'Windows':
    orig_path = r'RockingMotion/Journal/ESDB_dataset'
    load_model_path = r'RockingMotion/Journal/src'
else:
    load_model_path = r'C:\Users\rdasilv2\Gdrive\Backup Rafael\Documents\NC State\Research Related\ML and DSP\Proj - Body Rocking\src\transferlearning_ESDB_equipped_3'
    orig_path = r'C:\Users\rdasilv2\Gdrive\Backup Rafael\Documents\NC State\Research Related\ML and DSP\Proj - Body Rocking\EDAQA_dataset'
    output_study1 = r'C:\Users\rdasilv2\Gdrive\Backup Rafael\Documents\NC State\Research Related\ML and DSP\Proj - Body Rocking\Study1'
    output_study2 = r'C:\Users\rdasilv2\Gdrive\Backup Rafael\Documents\NC State\Research Related\ML and DSP\Proj - Body Rocking\Study2'

data_file_name = 'torso_100_mw.mat'
data_study1, data_study2, labels_study1, labels_study2 = load_session_EDAQA(orig_path)

samplingFreq = [60, 90]
overlap = 10

FSAMP = [60, 90]
Low = 0.05
High = [27, 40.5]
FONDOSCALA_ACC = 100
FONDOSCALA_GYR = 2000

accCoeff = FONDOSCALA_ACC*4/32768.0
gyrCoeff = FONDOSCALA_GYR/32768.0
magCoeff = 0.007629

filter_segment_and_save(data_study1, labels_study1, output_study1, FSAMP[0], High[0], samplingFreq[0])
filter_segment_and_save(data_study2, labels_study2, output_study2, FSAMP[1], High[1], samplingFreq[1])

