import tensorflow as tf
import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.regularizers import l2
import numpy as np
import collections


def bayesian_categorical_crossentropy(logits, nb_classes):
    def bayesian_categorical_crossentropy_internal(true, pred_var):
        mc_results = []
        for i in range(20):
            s = tf.random.normal(tf.shape(logits), mean=0.0, stddev=1.0)
            std_samples = tf.multiply(pred_var, s)
            distorted_loss = \
                tf.nn.softmax_cross_entropy_with_logits(logits=(logits + std_samples), labels=true)
            mc_results.append(distorted_loss)
        mc_results = tf.stack(mc_results, axis=0)
        var_loss = tf.reduce_mean(mc_results, axis=0)

        return var_loss

    return bayesian_categorical_crossentropy_internal


def CaliNet(fea_dim):
    opt_var = keras.optimizers.Adam(lr=1e-3)
    inputs = Input(shape=(fea_dim,))
    x = Dense(32, kernel_regularizer=l2(1e-5),
              bias_regularizer=l2(1e-5), activation='tanh')(inputs)
    x = Dense(64, kernel_regularizer=l2(1e-5),
              bias_regularizer=l2(1e-5), activation='tanh')(x)
    predictions = Dense(1, kernel_regularizer=l2(1e-5),
                        bias_regularizer=l2(1e-5), activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=opt_var,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def cal_bnn_vars(softmax_T, datavar_T, nb_classes):
    softmax_mean = np.mean(softmax_T, axis=0)
    datavar = np.mean(datavar_T, axis=0)
    datavar_std = np.std(datavar_T, axis=0)

    ratio_var = np.zeros((nb_classes,))
    pred_T = np.argmax(softmax_T, axis=-1)
    b = collections.Counter(pred_T)

    for key in b:
        ratio_var[key] = float(b[key]) / pred_T.shape[0]

        entropy_var = -1 * np.sum(np.multiply(np.log(np.finfo(float).eps \
                                                     + softmax_mean), softmax_mean), axis=-1)
        sum_c_mean_T = np.mean(np.sum(np.multiply(np.log(np.finfo(float).eps \
                                                         + softmax_T), softmax_T), axis=-1), axis=0)
        mi_var = entropy_var + sum_c_mean_T + np.finfo(float).eps

    return np.squeeze(ratio_var), np.squeeze(entropy_var), np.squeeze(mi_var), \
           np.squeeze(datavar), np.squeeze(datavar_std), np.squeeze(softmax_mean)


def MC_prediction(model, n_class, data, d):
    bs = data.shape[0]
    datavar_array = np.zeros((bs, 1))
    datavar_std_array = np.zeros((bs, 1))
    mi_var_array = np.zeros((bs, 1))
    entropy_var_array = np.zeros((bs, 1))
    ratio_var_array = np.zeros((bs, n_class))
    pred_probs_array = np.zeros((bs, n_class))

    for i in range(bs):
        mc_data = np.concatenate([data[i:i + 1, :, :] for mc in range(50)], axis=0)

        res = model.predict(mc_data)
        ratio_var, entropy_var, mi_var, datavar, datavar_std, softmax_mean = cal_bnn_vars(res[1], res[0],
                                                                                          n_class)
        pred_probs_array[i, :] = softmax_mean
        ratio_var_array[i, :] = ratio_var
        entropy_var_array[i] = entropy_var
        mi_var_array[i, :] = mi_var
        datavar_array[i, :] = datavar
        datavar_std_array[i, :] = datavar_std

    d['softmax'] = pred_probs_array
    d['entropy_var'] = entropy_var_array
    d['mi_var'] = mi_var_array
    d['data_var'] = datavar_std_array

    return d
