
'''
Code originally written by:

@author: RadModel Mohammadian Rad <Email: RadModel@fbk.eu>
Paper: "Stereotypical Motor Movement Detection in Dynamic Feature Space"
To reproduce the results of the Static-Features-Unbalanced and Static-Features-Balanced experiments on the simulated data.

The code was adapted and expanded to the paper:
""
by Rafael Luiz da Silva, email: rafaelonwork@gmail.com
This version runs 2 different models in EDAQA and ESDB dataset. For expanding WiderNet, please change number of filters
parameters in the code itself starting at line 353

'''

import pickle
import numpy as np
import scipy.io as sio
from sklearn.metrics import f1_score,  auc, roc_curve, roc_auc_score, recall_score, precision_score, matthews_corrcoef
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.core import Activation, Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution1D, Convolution2D, AveragePooling2D
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Model
from tensorflow.python.keras.regularizers import l2
from keras.utils import np_utils
from deeppy import StandardScaler
import keras
from keras import backend as K
import tensorflow as tf
#tf.python.control_flow_ops = tf
import platform
import collections
import os
import sys
import bayesian_utils
from scipy.ndimage import zoom
import copy


def keep_pos_exmpls_in_test_partitions(test_data, test_labels, split):

    # -------------------------------------------------------------
    # Prepare test data for SVM to be as close as possible to 50-50
    # -------------------------------------------------------------
    test_labels_pos_bin = test_labels == np.unique(test_labels)[1]
    test_labels_neg_bin = test_labels == np.unique(test_labels)[0]

    test_labels_pos = test_labels[test_labels_pos_bin]
    test_labels_neg = test_labels[test_labels_neg_bin]
    test_data_pos = test_data[test_labels_pos_bin[:, 0], :]
    test_data_neg = test_data[test_labels_neg_bin[:, 0], :]

    test_labels_firstpart = np.concatenate((test_labels_pos[:int(test_labels_pos.shape[0] / 2)],
                                            test_labels_neg[:int(split/2 - test_labels_pos.shape[0] / 2)]), axis=0)
    test_labels_2ndpart = np.concatenate((test_labels_pos[int(test_labels_pos.shape[0] / 2):],
                                          test_labels_neg[int(split/2 - test_labels_pos.shape[0] / 2):]), axis=0)

    test_data_firstpart = np.concatenate((test_data_pos[:int(test_data_pos.shape[0] / 2)],
                                          test_data_neg[:int(split/2 - test_labels_pos.shape[0] / 2)]), axis=0)
    test_data_2ndpart = np.concatenate((test_data_pos[int(test_data_pos.shape[0] / 2):],
                                        test_data_neg[int(split/2 - test_labels_pos.shape[0] / 2):]), axis=0)

    shuffle_indices = np.random.randint(0, test_labels_2ndpart.shape[0], test_labels_2ndpart.shape[0])
    test_labels_2ndpart = test_labels_2ndpart[shuffle_indices]
    test_labels = np.concatenate((test_labels_firstpart, test_labels_2ndpart), axis=0).reshape((-1, 1))
    test_data_2ndpart = test_data_2ndpart[shuffle_indices, :]
    test_data = np.concatenate((test_data_firstpart, test_data_2ndpart), axis=0)
    print('Values of labels in test 1st part {}'.format(np.unique(test_labels_firstpart)))
    print('Values of labels in test 2nd part {}'.format(np.unique(test_labels_2ndpart)))

    return test_data, test_labels


#----------------------------------------------------------
#                       PARAMETERS
#----------------------------------------------------------
# BNN approach
try:
  if sys.argv[1] is not None:
    BNN_ison = bool(int(sys.argv[1]))
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('BAYESIAN?: {}'.format(BNN_ison))
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
except:
    print('Nothing here')
    BNN_ison = True

# Choose model 0 - RadModel, 1 - WiderNet
model_id = 1
# 0 - ESDB dataset (1 subject, broken up into 7 session variables), 1 - EDAQA dataset (6 subjects), 2 - Simulated
choose_dataset = 0
#----------------------------------------------------------
#                       EXPERIMENT
#----------------------------------------------------------
if choose_dataset == 0:
    subNum = 7
    subjects = ["subject0", "subject1", "subject2", "subject3", "subject4", "subject5", "subject6"]
    print('ooooo Using ESDB Dataset ooooooo')
    if platform.system() != 'Windows':
        # set the output and input paths
        path = r'/path/tomylinux/ESDB_dataset'
        savePath = r'/path/tomylinux/Results/'  # save the results of unbalanced-CNN
    else:
        # set the output and input paths
        path = r'C:/path/tomywindows/ESDB_dataset'
        savePath = r'C:/path/tomywindows/Results/'  # save the results of unbalanced-CNN
elif choose_dataset == 1:
    subNum = 6
    subjects = ["subject0", "subject1", "subject2", "subject3", "subject4", "subject5"]
    print('ooooo Using EDAQA Dataset ooooooo')
    if platform.system() != 'Windows':
        # set the output and input paths
        path = r'/path/tomylinux/Study1_old'
        savePath = r'/path/tomylinux/Results/'  # save the results of unbalanced-CNN
    else:
        # set the output and input paths
        path = r'C:/path/tomywindows/Study1'
        savePath = r'C:/path/tomywindows/Results/'  # save the results of unbalanced-CNN
elif choose_dataset == 2:
    subNum = 5
    subjects = ["subject1", "subject2", "subject3", "subject4", "subject5"]
    print('ooooo Using Simulated Dataset ooooooo')
    if platform.system() != 'Windows':
        # set the output and input paths
        path = r'/path/tomylinux/Results/Intermediate_files/'
        savePath = r'/path/tomylinux/Results/'  # save the results of unbalanced-CNN
    else:
        # set the output and input paths
        path = r'/path/tomywindows/Results/Intermediate_files/'
        savePath = r'/path/tomywindows/Results/'  # save the results of unbalanced-CNN

print('ioioioioioioioioiioioioioioioioioioioioioioioioioioi')
print(path)
print('ioioioioioioioioiioioioioioioioioioioioioioioioioioi')
# Setting Network Parameters
nb_filters=[4, 4, 8]
hidden_neuron_num = 8
filter_size = [10, 1]
channels = 9
pool_size = (3,1)
strides=(2,1)

# Setting Learning Parameters
runNum = 10
nb_epoch = [35, 10]
learn_rate = 0.05
batch_size = 100
mtm=0.9
nb_classes=2
backend = 'tf'
borderMode = 'same'
balanced = 0 # unbalanced-CNN experiment.

f1Net = np.zeros([runNum, subNum])
accNet = np.zeros([runNum, subNum])
AUCNet = np.zeros([runNum, subNum])
precisionNet = np.zeros([runNum, subNum])
recallNet = np.zeros([runNum, subNum])
MCCNet = np.zeros([runNum, subNum])
fpr = dict()
tpr = dict()
roc_auc = dict()
study = 'Simulated_'
temp_X = []
temp_Y = []


# reading data
for i in range(np.shape(subjects)[0]):
    print(os.path.join(path, subjects[i] + '.mat'))
    matContent = sio.loadmat(os.path.join(path, subjects[i] + '.mat'))
    temp = matContent['X']
    temp_X.append(temp)
    temp = matContent['Labels']
    temp_Y.append(temp)

# creating training and test dataset based on one-subject-leave-out scenario.
for sub in range(subNum):
    old_testFeatures = temp_X[sub]
    testLabels = temp_Y[sub]
    testLabels = testLabels.astype(int).reshape((-1, 1))
    print('-----------------------------------------')
    print('Subject {}'.format(sub))
    print('-----------------------------------------')
    print('Values of labels in test {}'.format(np.unique(testLabels)))
    print('Share of classes: {}, {}'.format(
        sum(testLabels == np.unique(testLabels)[0]) /
        testLabels.shape[0],
        sum(testLabels == np.unique(testLabels)[1]) /
        testLabels.shape[0]))

    print('Test feat {}, Test labels {}'.format(old_testFeatures.shape, testLabels.shape))
    old_testFeatures, testLabels = keep_pos_exmpls_in_test_partitions(old_testFeatures, testLabels, old_testFeatures.shape[0])

    testLabels = np.squeeze(testLabels)

    train_index = np.setdiff1d(range(subNum), sub)
    old_trainingFeatures = temp_X[train_index[0]]
    trainingLabels = temp_Y[train_index[0]]
    train_index = np.setdiff1d(train_index, train_index[0])
    for j in range(len(train_index)):
        old_trainingFeatures = np.concatenate((old_trainingFeatures, temp_X[train_index[j]]), axis=0)
        trainingLabels = np.concatenate((trainingLabels, temp_Y[train_index[j]]), axis=1)

    print('Train feat {}, Train labels {}'.format(old_trainingFeatures.shape, trainingLabels.shape))

    # Reshape data to 100 samples when running EDAQA dataset so it is compatible with ESDB for Transfer learning
    if choose_dataset == 1:
        testFeatures = zoom(old_testFeatures, (1, 100/old_testFeatures.shape[1], 1))
        trainingFeatures = zoom(old_trainingFeatures, (1, 100/old_trainingFeatures.shape[1], 1))

    else:
        testFeatures = zoom(old_testFeatures, (1, 60/100, 1))
        trainingFeatures = zoom(old_trainingFeatures, (1, 60/100, 1))
    #     testFeatures = old_testFeatures
    #     trainingFeatures = old_trainingFeatures
    print('Reshaped to - Train feat {}, Test feat {}'.format(trainingFeatures.shape, testFeatures.shape))
    '''
    if balanced: # To replicate the results of CNN on balanced data or Static-Features-Balanced experiment
        t = np.sum(trainingLabels, axis=1)
        temp_smm = trainingFeatures[trainingLabels[0, :] == 1, :, :]
        temp_nosmm = trainingFeatures[trainingLabels[0, :] == 0, :, :]
        idx = np.random.permutation(temp_nosmm.shape[0])
        idx = idx[:t]
        temp_nosmm = temp_nosmm[idx, :, :]
        trainingFeatures = np.concatenate((temp_nosmm, temp_smm), axis=0)
        trainingLabels = np.zeros([1, 2*t])
        trainingLabels[0, t:]=1
        idx = np.random.permutation(trainingFeatures.shape[0])
        trainingFeatures = trainingFeatures[idx, :, :]
        trainingLabels = trainingLabels[0, idx]
    '''
    # constructing data compatible with tensorflow backend
    trainingFeatures = np.transpose(trainingFeatures, (0, 2, 1)) # Chanel should be in the second dimension
    print(trainingFeatures.shape)
    trainingFeatures = np.float64(trainingFeatures[..., np.newaxis]) #
    print(trainingFeatures.shape)
    trainingFeatures = np.transpose(trainingFeatures, (0, 2, 3, 1))
    print(trainingFeatures.shape)
    testFeatures = np.transpose(testFeatures, (0, 2, 1)) 
    testFeatures = np.float64(testFeatures[..., np.newaxis])
    testFeatures = np.transpose(testFeatures, (0, 2, 3,1))

    trainingLabels = trainingLabels.astype(int)
    trainingLabels = np.squeeze(trainingLabels)

    trainingLabels = np_utils.to_categorical(trainingLabels, nb_classes)
    testLabels1 = np_utils.to_categorical(testLabels, nb_classes)

    # Normalization
    scaler = StandardScaler()
    scaler.fit(trainingFeatures)
    trainingFeatures = scaler.transform(trainingFeatures)
    testFeatures = scaler.transform(testFeatures)

    for run in range(runNum):

        print('Starting run {}'.format(run))
        # Perform transfer learning
        try:
            if sys.argv[2] is not None:
                is_TL = bool(int(sys.argv[2]))
                print('******************************************************************')
                print('Transfer Learning: {}'.format(is_TL))
                print('******************************************************************')
        except:
            is_TL = False
            print('Nothing here for Transfer Learning, assuming False')

        # Model Load weights
        try:
            if sys.argv[3] is not None:
                load_ws = bool(int(sys.argv[3]))
                print('******************************************************************')
                print('Load model weights: {}'.format(load_ws))
                print('******************************************************************')
        except:
            load_ws = False
            print('Nothing here for model load weights, assuming False')

        # Model Save weights
        try:
            if sys.argv[4] is not None:
                save_ws = bool(int(sys.argv[4]))
                print('******************************************************************')
                print('Save model weights: {}'.format(save_ws))
                print('******************************************************************')
        except:
            save_ws = False
            print('Nothing here for model save weights, assuming False')

        if is_TL:
            perform_training = False
        else:
            perform_training = True

        if model_id == 0:

            # Prepare network inputs
            x_in = keras.layers.Input(shape=(trainingFeatures.shape[1], trainingFeatures.shape[2],
                                             trainingFeatures.shape[3]))

            h = keras.layers.Convolution2D(nb_filters[0], filter_size[0], filter_size[1], activation='relu',
                                           border_mode=borderMode, dim_ordering=backend, init='he_normal')(x_in)
            h = keras.layers.AveragePooling2D(pool_size, strides, border_mode=borderMode, dim_ordering=backend)(h)
            if BNN_ison:
                h = keras.layers.Dropout(0.05)(h, training=True)
            h = keras.layers.Convolution2D(nb_filters[1], filter_size[0], filter_size[1], activation='relu',
                                           border_mode=borderMode, dim_ordering=backend, init='he_normal', trainable=perform_training)(h)
            h = keras.layers.AveragePooling2D(pool_size, strides, border_mode=borderMode, dim_ordering=backend)(h)
            if BNN_ison:
                h = keras.layers.Dropout(0.05)(h, training=True)

            h = keras.layers.Convolution2D(nb_filters[2], filter_size[0], filter_size[1], activation='relu',
                                           border_mode=borderMode, dim_ordering=backend, init='he_normal', trainable=perform_training)(h)
            h = keras.layers.AveragePooling2D(pool_size, strides, border_mode=borderMode, dim_ordering=backend)(h)
            if BNN_ison:
                h = keras.layers.Dropout(0.05)(h, training=True)

            h = keras.layers.Flatten()(h)

            h = keras.layers.BatchNormalization(epsilon=1.1e-6, mode=0, momentum=0.9, weights=None)(h)

            h = keras.layers.Dense(8, init='he_normal')(h)
            h = keras.layers.Dropout(0.2)(h, training=True)
            h = keras.layers.Activation('relu')(h)

            logits = keras.layers.Dense(nb_classes, name='logits')(h)
            softmax = keras.layers.Activation('softmax', name='softmax')(logits)

            variance_pre = keras.layers.Dense(1, name='variance_input')(h)
            variance = keras.layers.Activation('softplus', name='logits_variance')(variance_pre)

            if BNN_ison:
                model = keras.models.Model(inputs=x_in, outputs=[variance, softmax])
            else:
                model = keras.models.Model(inputs=x_in, outputs=softmax)

        elif model_id == 1:
            # Number of filters 4,4,8 dense layer 8
            # Prepare network inputs
            print(trainingFeatures.shape)
            # Prepare network inputs
            x_in = keras.layers.Input(shape=(trainingFeatures.shape[1], trainingFeatures.shape[2],
                                             trainingFeatures.shape[3]))

            h = keras.layers.Convolution2D(64, 10, filter_size[1], activation='relu',
                                           border_mode=borderMode, dim_ordering=backend, init='he_normal')(x_in)
            h = keras.layers.AveragePooling2D(pool_size, strides, border_mode=borderMode, dim_ordering=backend)(h)
            if BNN_ison:
                h = keras.layers.Dropout(0.05)(h, training=True)
            h = keras.layers.Convolution2D(64, 10, filter_size[1], activation='relu',
                                           border_mode=borderMode, dim_ordering=backend, init='he_normal',
                                           trainable=perform_training)(h)
            h = keras.layers.AveragePooling2D(pool_size, strides, border_mode=borderMode, dim_ordering=backend)(h)
            if BNN_ison:
                h = keras.layers.Dropout(0.05)(h, training=True)

            h = keras.layers.Convolution2D(128, 10, filter_size[1], activation='relu',
                                           border_mode=borderMode, dim_ordering=backend, init='he_normal', trainable=perform_training)(h)
            h = keras.layers.AveragePooling2D(pool_size, strides, border_mode=borderMode, dim_ordering=backend)(h)
            if BNN_ison:
                h = keras.layers.Dropout(0.05)(h, training=True)

            h = keras.layers.Flatten()(h)

            h = keras.layers.BatchNormalization(epsilon=1.1e-6, mode=0, momentum=0.9, weights=None)(h)

            h = keras.layers.Dense(128, init='he_normal')(h)
            h = keras.layers.Dropout(0.2)(h, training=True)
            h = keras.layers.Activation('relu')(h)

            logits = keras.layers.Dense(nb_classes, name='logits')(h)
            softmax = keras.layers.Activation('softmax', name='softmax')(logits)

            variance_pre = keras.layers.Dense(1, name='variance_input')(h)
            variance = keras.layers.Activation('softplus', name='logits_variance')(variance_pre)

            if BNN_ison:
                model = keras.models.Model(inputs=x_in, outputs=[variance, softmax])
            else:
                model = keras.models.Model(inputs=x_in, outputs=softmax)

        EearlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')

        if BNN_ison:
            model = keras.models.Model(inputs=x_in, outputs=[variance, softmax])
            cali_net = bayesian_utils.CaliNet(3)
        else:
            model = keras.models.Model(inputs=x_in, outputs=softmax)
        # model.summary()
        for ep in range(len(nb_epoch)):

            sgd=SGD(lr=learn_rate/10**ep, momentum=0.9, decay=0.03, nesterov=True)
            global_step = tf.Variable(0, trainable=False, name="global_step")
            lr_intent = tf.compat.v1.train.exponential_decay(1e-4, global_step, 1000, 0.995, staircase=True)
            opt = tf.compat.v1.train.AdamOptimizer(lr_intent)
            if BNN_ison:

                model.compile(loss={'logits_variance': bayesian_utils.bayesian_categorical_crossentropy(logits, 2)},
              metrics={'logits_variance': keras.metrics.categorical_accuracy},
              optimizer=opt)
            else:
                model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

            model.summary()

            if load_ws:
                model.load_weights('model_' + str(BNN_ison) + '.h5')
                print('Successfully loaded model weights.')

            model.fit(trainingFeatures, trainingLabels, batch_size=batch_size, nb_epoch=nb_epoch[ep],
             verbose=2, callbacks=[EearlyStopping], validation_split=0.1)

            if save_ws:
                model.save_weights('model_' + str(BNN_ison) + '.h5')
                print('Successfully saved model weights.')

        print('Sucessfully trained CNN.')

        # ---------------------------------------------------------------------------------------
        #         Prediction and Bayes inference
        # ---------------------------------------------------------------------------------------
        val_outcome = {}
        test_outcome = {}
        valtest_split = int(testFeatures.shape[0]/2)
        valX = testFeatures[valtest_split:, :]
        valLabels = testLabels[valtest_split:].reshape(-1, 1)
        testX = testFeatures[:valtest_split, :]
        testXLabels = testLabels[:valtest_split].reshape(-1, 1)
        print('-----------------------------')
        print('{}, {}, {}'.format(testFeatures.shape, valLabels.shape, testLabels.shape))
        if BNN_ison:
            val_outcome = bayesian_utils.MC_prediction(model, nb_classes, valX, val_outcome)
            test_outcome = bayesian_utils.MC_prediction(model, nb_classes, testX, test_outcome)
        else:
            results = model.predict(testX)

        if BNN_ison:
            predict = np.array(np.argmax(val_outcome['softmax'], axis=1))
            val_outcome['predictions'] = predict
            val_outcome['correct'] = predict.ravel() == valLabels.ravel()
            predict = np.array(np.argmax(test_outcome['softmax'], axis=1))
            test_outcome['predictions'] = predict
            test_outcome['correct'] = predict.ravel() == testXLabels.ravel()
            # Train CaliNet on validation data
            val_uncertainties = np.concatenate(
                (val_outcome['data_var'], val_outcome['entropy_var'], val_outcome['mi_var']), axis=-1)
            cali_net.fit(val_uncertainties, val_outcome['correct'], batch_size=32, epochs=1)
            # Test CaliNet on test data
            test_uncertainties = np.concatenate(
                (test_outcome['data_var'], test_outcome['entropy_var'], test_outcome['mi_var']), axis=-1)
            test_cali_prob = cali_net.predict(test_uncertainties)
            # cali_net.save(os.path.join(os.path.expanduser('~'), savePath, 'calinet_except_subj_' + str(sub) + '_' + str(subNum)))

            # Adapted from legacy code
            predicted_labelsNet = test_outcome['predictions'].reshape(-1, 1)
            soft_targets_train = model.predict(trainingFeatures)[1]
            soft_targets_test = test_outcome['softmax']
        else:
            predict = np.array(np.argmax(results, axis=1))
            val_outcome['predictions'] = np.array([0, 0])
            val_outcome['correct'] = np.array([0, 0])
            test_outcome['predictions'] = np.array([0, 0])
            test_outcome['correct'] = np.array([0, 0])
            val_outcome['softmax'] = np.array([0, 0])
            test_uncertainties = np.array([0, 0])
            val_uncertainties = np.array([0, 0])
            test_cali_prob = np.array([0, 0])
            # legacy code
            predicted_labelsNet = np.float64(model.predict(testFeatures), verbose=0)
            soft_targets_train = model.predict(trainingFeatures)
            soft_targets_test = model.predict(testFeatures)
            test_outcome['softmax'] = soft_targets_test
            predicted_labelsNet = np.argmax(predicted_labelsNet, axis=1).reshape(-1, 1)
            testXLabels = testLabels.reshape(-1, 1)

        # Saving the learned features from the flattening layer
        if not is_TL:
            print('Starting output features since TL is on')
            if model_id == 0:
                dim = 9
            elif model_id == 1:
                dim = 10

            print(model.layers[0].input)
            print(model.layers[dim].output)
            print(K.learning_phase())

            get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[dim].output, ])
            layer_output_training = get_3rd_layer_output([trainingFeatures])[0]
            layer_output_test = get_3rd_layer_output([testFeatures])[0]
            if BNN_ison:
                layer_output_val = get_3rd_layer_output([valX])[0]
            else:
                layer_output_val = 0
            # sio.savemat(os.path.join(os.path.expanduser('~'), savePath, 'CNN_learned_features' + '_sub_'+str(sub+1) +'_run_'+ str(run+1)+ '_' + str(BNN_ison) + '_' + '.mat'), {'trainingFeatures':layer_output_training, 'trainingLabels':trainingLabels,
            #    'testFeatures':layer_output_test, 'testLabels':testLabels, 'valFeatures':layer_output_val, 'valLabels':valLabels})
        else:
            print('No output features since TL is on')
        # evaluation
        try:
            precisionNet[run, sub] = precision_score(testXLabels, predicted_labelsNet)
            recallNet[run, sub] = recall_score(testXLabels, predicted_labelsNet)
            f1Net[run, sub] = f1_score(testXLabels, predicted_labelsNet)
            MCCNet[run, sub] = matthews_corrcoef(testXLabels, predicted_labelsNet)
            print('Subject %d : Run %d :F1_Score_Net: %.4f' % (sub+1, run+1, f1Net[run, sub]))
            AUCNet[run, sub] = roc_auc_score(testXLabels, soft_targets_test[:, 1])
            print('Subject %d : Run %d :AUC_Net: %.4f' % (sub+1, run+1, AUCNet[run, sub]))
            fpr[run, sub], tpr[run, sub], _ = roc_curve(testXLabels, soft_targets_test[:, 1])
            roc_auc[run, sub] = auc(fpr[run, sub], tpr[run, sub])
        except:
            fpr[run, sub] = 0
            tpr[run, sub] = 0
            print('Bad values for scores boy...')

        try:
            print('Share of classes ouputs: {}, {}'.format(
                sum(predicted_labelsNet == np.unique(predicted_labelsNet)[0]) / predicted_labelsNet.shape[0],
                sum(predicted_labelsNet == np.unique(predicted_labelsNet)[1]) / predicted_labelsNet.shape[0]))
        except:
            print('Bad predictions here boy...')

        print('Values of labels in test {}'.format(np.unique(predicted_labelsNet)))
        print('Share of classes: {}, {}'.format(
            sum(testXLabels == np.unique(testXLabels)[0]) /
            testXLabels.shape[0],
            sum(testXLabels == np.unique(testXLabels)[1]) /
            testXLabels.shape[0]))

        # # save the model and weights
        # json_string = model.to_json()
        # open(savePath + study + 'CNN' + str(sub+1) + '_Run_' + str(run+1) + '.json', 'w').write(json_string)
        # model.save_weights(savePath + study + 'CNN' + str(sub+1) + '_Run_' + str(run+1) + '.h5', overwrite=True)
        # # save results
        # sio.savemat(savePath + study + 'CNN_Results' + '.mat', {'precisionNet': precisionNet,
        # 'recallNet': recallNet, 'f1Net': f1Net, 'AUCNet': AUCNet, 'MCCNet': MCCNet})

        # ---------------------------------------------------------------------------------------
        #         Save data
        # ---------------------------------------------------------------------------------------
        print('Test Labels {}'.format(testXLabels.shape))
        print('Predictions {}'.format(predicted_labelsNet.shape))
        print('Uncertainties {}'.format(test_uncertainties.shape))
        print('Cal Probs {}'.format(test_cali_prob.shape))
        print('Sfx: {}'.format(test_outcome['softmax'].shape))
        # sio.savemat(os.path.join(os.path.expanduser('~'), savePath,
        #                          'data_subj' + str(sub) + '_out_baye' + str(BNN_ison) + '_' + str(run) + '.mat'), {
        sio.savemat('data_subj' + str(sub) + '_out_baye' + str(BNN_ison) + '_' + str(run) + '.mat', {
            'test_correct_all': test_outcome['correct'],
            'test_cali_prob_all': test_cali_prob,
            'test_uncertainties_all': test_uncertainties,
            'test_pred_svm': predicted_labelsNet,
            'svm_softmax': test_outcome['softmax'],
            'test_softmax_all': test_outcome['softmax'],
            'test_gt': testXLabels,
            'test_data_all': 0,
            'train_pred': 0,
            'train_softmax_all': 0,
            'train_gt': 0,
            'train_data_all': 0,
            'train_idx': 0,
            'train_loss': 0,
            'val_gt_all': valLabels,
            'val_uncertainties_all': val_uncertainties,
            'val_correct_all': val_outcome['correct'],
            'val_result_all': 0,
            'val_softmax_all': val_outcome['softmax'],
            'val_data_all': 0,
            'test_sizes': 0,
            'val_sizes': 0,
            'track_pred': 0,
            'track_softmax_all': 0,
            'track_gt': 0,
            'track_data_all': 0,
            'track_uncertainties_all': 0,
            'roc_x': fpr[run, sub],
            'roc_y': tpr[run, sub]
        })
# pickle.dump(fpr, open(savePath + 'CNN_pickle_fpr.p', "wb"))
# pickle.dump(tpr, open(savePath + 'CNN_pickle_tpr.p', "wb"))
# pickle.dump(roc_auc, open(savePath + 'CNN_pickle_roc.p', "wb"))

