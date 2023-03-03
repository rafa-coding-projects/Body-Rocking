import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score, \
    matthews_corrcoef, cohen_kappa_score, precision_score, recall_score, confusion_matrix as cm
from sklearn.calibration import calibration_curve

import scipy.io as sio
import platform
import copy
import os
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import pickle
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import sys


def create_twin_plot(x, y1, y2, labels):

    ''''
    :param
    x: commons horizontal axis between y1 and y2
    y1: first observable variable
    y2: second observable variable
    labels: 4 element string vector containing labels for x, y1, y2 and figure filename
    '''

    fig, ax1 = plt.subplots()
    y1 = np.nan_to_num(y1)
    y2 = np.nan_to_num(y2)
    try:
        idx_min = np.where(np.abs(y1 - y2) == np.amin(np.abs(y1 - y2)))[0].item()
    except:
        try:
            idx_min = np.where(np.abs(y1 - y2) == np.amin(np.abs(y1 - y2)))[0][0]
        except:
            print('im here')

    # For cases other than sensitivity and specificity
    if x[idx_min] > 0.95:
        try:
            # If maximum is in a plateau or mutlitple maxima are found
            if len(np.where(y1 == np.amax(y1))[0]) > 1 or len(np.where(y2 == np.amax(y2))[0]) > 1:
                val1 = np.where(y1 == np.amax(y1))[0][0]
                val2 = np.where(y2 == np.amax(y2))[0][0]
            else:
                val1 = np.where(y1 == np.amax(y1))[0].item()
                val2 = np.where(y2 == np.amax(y2))[0].item()
        except:
            print('Problem with array dimensions')
        try:
            if val1 != val2:
                diff = np.abs(val1 - val2)
                # Find index closed to the min of maximums plus the median between maximums
                idx_min = int(np.rint(np.minimum(val1, val2) + diff/2))
            else:
                idx_min = val1
        except:
            print('noooooooooooooooo')

    color = 'tab:red'
    ax1.set_xlabel(labels[0])
    ax1.set_ylabel(labels[1], color=color)
    ax1.plot(x, y1, color=color)
    ax1.tick_params(axis=labels[1], labelcolor=color, labelsize=25)
    ax1.set_ylim(0, 1)
    try:
        ax1.plot(x[idx_min], y1[idx_min], marker='o', markersize=5, color='purple')
    except:
        print('again')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(labels[2], color=color)  # we already handled the x-label with ax1
    ax2.plot(x, y2, color=color)
    ax2.tick_params(axis=labels[2], labelcolor=color, labelsize=25)
    try:
        ax1.vlines(x=x[idx_min], ymin=0, ymax=1, linestyles='dashed',
                   colors='purple')
        ax1.annotate('Optimal Threshold = {:0.2f}'.format(x[idx_min]), xy=(x[idx_min], y1[idx_min]), xycoords='data',
                    xytext=(x[idx_min]+.05, 0.2), textcoords='axes fraction', color='purple')
    except:
        print('Not possible to draw a line for {}, idx value {}'.format(labels[3], idx_min))

    ax2.set_ylim(0, 1)
    ax1.plot(x[idx_min], y2[idx_min], marker='o', markersize=5, color='purple')
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    #mng.window.showMaximized()
    #mng.window.state('zoomed')
    plt.show(block=False)
    plt.pause(0.001)
    #manager = plt.get_current_fig_manager()
    #manager.window.showMaximized()
    #manager = plt.get_current_fig_manager()
    #manager.resize(*manager.window.maxsize())
    try:
        fig.savefig(os.path.join(orig_path, 'results', 'plot_twins', labels[3] + '.png'))
        fig.savefig(os.path.join(orig_path, 'results', 'plot_twins', labels[3] + '.eps'))
    except:
        os.makedirs(os.path.join(orig_path, 'results', 'plot_twins'))
    plt.close(fig)


def compute_over_runs(dictionary, y_gt, y_logits, sbj):

    fpr, tpr, _ = roc_curve(y_gt, y_logits, pos_label=1, drop_intermediate=True)

    dictionary['aggregate_auc'][exec_run, sbj] = auc(fpr, tpr)
    dictionary['aggregate_tpr'][exec_run, :, sbj] = np.interp(mean_fpr, fpr, tpr)
    dictionary['aggregate_tpr'][exec_run, 0, sbj] = 0.0
    idx = 0
    for thr in np.linspace(0, 1, 100, endpoint=True):
        y_thr = y_logits >= thr
        y_thr = y_thr.astype('int64')
        dictionary['aggregate_perc'][exec_run, idx, sbj] = 100 * np.sum(y_thr).astype('int') / np.sum(y_gt).astype('int')
        dictionary['aggregate_mcc'][exec_run, idx, sbj] = matthews_corrcoef(y_gt, y_thr)
        dictionary['aggregate_ck'][exec_run, idx, sbj] = cohen_kappa_score(y_gt, y_thr)
        dictionary['aggregate_f1'][exec_run, idx, sbj] = f1_score(y_gt, y_thr, zero_division='warn')

        try:
            dictionary['aggregate_tn'][exec_run, idx, sbj], dictionary['aggregate_fp'][exec_run, idx, sbj], \
            dictionary['aggregate_fn'][exec_run, idx, sbj], dictionary['aggregate_tp'][exec_run, idx, sbj] = \
                cm(y_gt, y_thr).ravel()
            dictionary['specificity'][exec_run, idx, sbj] = np.divide(dictionary['aggregate_tn'][exec_run, idx, sbj],
                                                                      (dictionary['aggregate_tn'][exec_run, idx, sbj]
                                                                       + dictionary['aggregate_fp'][
                                                                           exec_run, idx, sbj]))
        except:
            print('Confusion Matrix not possible to be computed, no distinct labels')
            dictionary['aggregate_tn'][exec_run, idx, sbj], dictionary['aggregate_fp'][exec_run, idx, sbj], \
            dictionary['aggregate_fn'][exec_run, idx, sbj], dictionary['aggregate_tp'][exec_run, idx, sbj] = \
                0, 0, 0, 0
            dictionary['specificity'][exec_run, idx, sbj] = 0

        dictionary['sensitivity'][exec_run, idx, sbj] = recall_score(y_gt, y_thr, zero_division='warn')
        dictionary['precision'][exec_run, idx, sbj] = precision_score(y_gt, y_thr, zero_division='warn')

        idx += 1

    return dictionary


def iterate_and_plot(list_of_signals, list_of_labels, list_of_axis):

    ''''
    Iterate over list of signals, plot them, and save in .png and .eps formats
    list_of_signals: signals to be plotted
    list_of_labels: name associated with signals in list_of_signals
    list_of_axis: list with axis labels, where list_of_axis[0] is the x axis and list_of_axis[1] is the y axis,
    and list_of_axis[2] is the filename with no extension
    '''
    fig, ax = plt.subplots()

    for i, j in zip(list_of_signals, list_of_labels):
        x = np.linspace(0, 1, i.shape[0])
        ax.plot(x, i, label=j)

    ax.legend(fontsize=25)
    ax.set_xlabel(list_of_axis[0])
    ax.set_xlabel(list_of_axis[1])
    ax.tick_params(axis='both', which='major', labelsize=30)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    #mng.window.showMaximized()
    mng.window.state('zoomed')
    plt.show(block=False)
    plt.pause(0.001)
    #manager = plt.get_current_fig_manager()
    #manager.window.showMaximized()
    #manager = plt.get_current_fig_manager()
    #manager.resize(*manager.window.maxsize())

    try:
        fig.savefig(os.path.join(orig_path, 'results', 'plot_all_metrics', list_of_axis[2] + '.png'))
        fig.savefig(os.path.join(orig_path, 'results', 'plot_all_metrics', list_of_axis[2] + '.eps'))
    except:
        os.makedirs(os.path.join(orig_path, 'results', 'plot_all_metrics'))
        fig.savefig(os.path.join(orig_path, 'results', 'plot_all_metrics', list_of_axis[2] + '.png'))
        fig.savefig(os.path.join(orig_path, 'results', 'plot_all_metrics', list_of_axis[2] + '.eps'))

    plt.close(fig)


def generate_twin_plots_for_each_subject(dictionary, iter_method, subject, approach):

    # Generate plots for pairs of metrics per subject

    sp_mean, ss_mean = np.mean(dictionary['specificity'][:, :, subject], axis=iter_method), \
                       np.mean(dictionary['sensitivity'][:, :, subject], axis=iter_method)
    if len(sp_mean) > 0 or len(ss_mean) > 0:
        create_twin_plot(np.linspace(0, 1, 100, endpoint=True), sp_mean, ss_mean,
                     ['Threshold', 'Specificity', 'Sensitivity',
                      'subj_' + str(subject + 1) + 'thr_spec_sens_' + approach])
    else:
        print('Not enough sensitivity points for subject {}'.format(subject))

    f1_mean, mcc_mean = np.mean(dictionary['aggregate_f1'][:, :, subject], axis=iter_method), \
                        np.mean(dictionary['aggregate_mcc'][:, :, subject], axis=iter_method)
    if len(f1_mean) > 0 or len(mcc_mean):
        create_twin_plot(np.linspace(0, 1, 100, endpoint=True), f1_mean, mcc_mean,
                     ['Threshold', 'F1', 'MCC', 'subj_' + str(subject + 1) + 'thr_f1_mcc_' + approach])
    else:
        print('Not enough F1 or MCC points for subject {}'.format(subject))

    f1_mean, ck_mean = np.mean(dictionary['aggregate_f1'][:, :, subject], axis=iter_method), \
                       np.mean(dictionary['aggregate_ck'][:, :, subject], axis=iter_method)
    if len(f1_mean) > 0 or len(ck_mean):
        create_twin_plot(np.linspace(0, 1, 100, endpoint=True), f1_mean, ck_mean,
                     ['Threshold', 'F1', 'Cohen-Kappa', 'subj_' + str(subject + 1) + 'thr_f1_ck_' + approach])
    else:
        print('Not enough F1 or CK points for subject {}'.format(subject))

    mcc_mean, ck_mean = np.mean(dictionary['aggregate_mcc'][:, :, subject], axis=iter_method), \
                        np.mean(dictionary['aggregate_ck'][:, :, subject], axis=iter_method)
    if len(mcc_mean) > 0 or len(ck_mean):
        create_twin_plot(np.linspace(0, 1, 100, endpoint=True), f1_mean, ck_mean,
                     ['Threshold', 'MCC', 'Cohen-Kappa', 'subj_' + str(subject + 1) + 'thr_mcc_ck_' + approach])
    else:
        print('Not enough MCC or CK points for subject {}'.format(subject))

    return [sp_mean, ss_mean, f1_mean, mcc_mean, ck_mean], ['Specificity', 'Sensitivity', 'F1', 'MCC',
                                                            'Cohen-Kappa']


def generate_twin_plot_over_all_subjects(dictionary, iter_method, filetitle, approach):

    # Generate plots for pairs of metrics average over all subject
    sp_mean, ss_mean = np.mean(dictionary['specificity'], axis=iter_method), \
                       np.mean(dictionary['sensitivity'], axis=iter_method)
    if len(sp_mean) > 0 or len(ss_mean) > 0:
        create_twin_plot(np.linspace(0, 1, 100, endpoint=True), sp_mean, ss_mean,
                     ['Threshold', 'Specificity', 'Sensitivity',
                      filetitle + '_thr_spec_sens_' + approach])
    else:
        print('Not enough sensitivity points for all subjects')

    f1_mean, mcc_mean = np.mean(dictionary['aggregate_f1'], axis=iter_method), \
                        np.mean(dictionary['aggregate_mcc'], axis=iter_method)
    if len(f1_mean) > 0 or len(mcc_mean) > 0:
        create_twin_plot(np.linspace(0, 1, 100, endpoint=True), f1_mean, mcc_mean,
                     ['Threshold', 'F1', 'MCC', filetitle + '_thr_f1_mcc_' + approach])
    else:
        print('Not enough F1 or MCC points for all subejcts')

    f1_mean, ck_mean = np.mean(dictionary['aggregate_f1'], axis=iter_method), \
                       np.mean(dictionary['aggregate_ck'], axis=iter_method)
    if len(f1_mean) > 0 or len(ck_mean) > 0:
        create_twin_plot(np.linspace(0, 1, 100, endpoint=True), f1_mean, ck_mean,
                     ['Threshold', 'F1', 'Cohen-Kappa', filetitle + '_thr_f1_ck_' + approach])
    else:
        print('Not enough F1 or CK points for all subejects')

    mcc_mean, ck_mean = np.mean(dictionary['aggregate_mcc'], axis=iter_method), \
                        np.mean(dictionary['aggregate_ck'], axis=iter_method)
    if len(mcc_mean) > 0 or len(ck_mean) > 0:
        create_twin_plot(np.linspace(0, 1, 100, endpoint=True), f1_mean, ck_mean,
                     ['Threshold', 'MCC', 'Cohen-Kappa', filetitle + '_thr_mcc_ck_' + approach])
    else:
        print('Not enough MCC or CK points for all subejcts')

    agg_perc_mean = np.mean(dictionary['aggregate_perc'], axis=iter_method)

    return [sp_mean, ss_mean, f1_mean, mcc_mean, ck_mean, agg_perc_mean], ['Specificity', 'Sensitivity', 'F1', 'MCC',
                                                            'Cohen-Kappa', 'Activity %']


def plot_histograms(agg_unc, texts):
    '''
    agg_unc: aggregated list results for uncertainty
    agg_calprob: aggregated list results for calibrated probability
    texts: list of labels: 0 = uncertainty, 1 = Cal. Prob., 2 = file suffix
    '''

    figH, axH = plt.subplots()
    axH.hist(agg_unc, label=texts[:2])
    axH.legend(fontsize=40)
    axH.set_ylabel('Occurrences', fontsize=40)
    axH.set_xlabel('Values', fontsize=40)
    axH.tick_params(axis='both', which='major', labelsize=25)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    #mng.window.showMaximized()
    mng.window.state('zoomed')
    plt.show(block=False)
    plt.pause(0.001)
    # manager = plt.get_current_fig_manager()
    # manager.window.showMaximized()
    try:
        figH.savefig(os.path.join(orig_path, 'results', 'plot_histograms', 'hist_' + texts[2] + '.png'))
        figH.savefig(os.path.join(orig_path, 'results', 'plot_histograms', 'hist_' + texts[2] + '.eps'))
    except:
        os.makedirs(os.path.join(orig_path, 'results', 'plot_histograms'))
        figH.savefig(os.path.join(orig_path, 'results', 'plot_histograms', 'hist_' + texts[2] + '.png'))
        figH.savefig(os.path.join(orig_path, 'results', 'plot_histograms', 'hist_' + texts[2] + '.eps'))

    plt.close(figH)


def reliability_diagram(prob_pos, y_test, text, file_suffix):

    def concatenate(list_of_arrays):


        concatentated_array = copy.deepcopy(list_of_arrays[0])

        for i in list_of_arrays[1:]:
            concatentated_array = np.concatenate((concatentated_array, i), axis=0)

        return concatentated_array

    def integrate(y1, y2, x):
        '''
        Extracted from
        https://stackoverflow.com/questions/25439243/find-the-area-between-two-curves-plotted-in-matplotlib-fill-between-area
        Thanks to StackOverflow, user VBB!
        '''
        z = y1 - y2
        dx = x[1:] - x[:-1]
        cross_test = np.sign(z[:-1] * z[1:])
        dx_intersect = - dx / (z[1:] - z[:-1]) * z[:-1]
        areas_pos = abs(z[:-1] + z[1:]) * 0.5 * dx  # signs of both z are same
        areas_neg = 0.5 * dx_intersect * abs(z[:-1]) + 0.5 * (dx - dx_intersect) * abs(z[1:])
        areas = np.where(cross_test < 0, areas_neg, areas_pos)
        total_area = np.sum(areas)

        return total_area

    prob_pos_all = concatenate(prob_pos)
    y_test_all = concatenate(y_test)
    text.append('All subjects')
    prob_pos.append(prob_pos_all)
    y_test.append(y_test_all)

    linewidth_for_plots = 1
    style = 's-'
    fig = plt.figure()
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    for p, truth, txt in zip(prob_pos, y_test, text):

        # p = (p - p.min()) / (p.max() - p.min())

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(truth, p, n_bins=10)

        if txt is 'All subjects':
            linewidth_for_plots = 3
            style = 'D--'
            try:
                sio.savemat(os.path.join(orig_path, 'results', 'reliability_diagram', 'rd_var_all_subjs.mat'),
                        {'mean_predicted_value': mean_predicted_value,
                         'fraction_of_positives': fraction_of_positives,
                         'p': p})
            except:
                os.makedirs(os.path.join(orig_path, 'results', 'reliability_diagram'))
                sio.savemat(os.path.join(orig_path, 'results', 'reliability_diagram', 'rd_var_all_subjs.mat'),
                        {'mean_predicted_value': mean_predicted_value,
                         'fraction_of_positives': fraction_of_positives,
                         'p': p})

        ax1.plot(mean_predicted_value, fraction_of_positives, style,
                 label="%s" % (txt, ), linewidth=linewidth_for_plots)

        ax2.hist(p, range=(0, 1), bins=10, label=txt,
                 histtype="step", lw=linewidth_for_plots)

    perf_calibrated = np.linspace(0, 1, 10)
    reference = np.linspace(min(mean_predicted_value), max(mean_predicted_value), 10)
    ax1.plot(perf_calibrated, perf_calibrated, "k:", label="Perfectly calibrated")
    y = np.interp(reference, mean_predicted_value, fraction_of_positives)
    ax1.fill_between(reference, reference, y, color='lightcoral',
                        label=r'Area = {:.2f}'.format(abs(auc(reference, y) - auc(reference, reference))))
    # ax1.fill_between(reference, reference, fraction_of_positives, color='lightcoral',
    #                     label=r'Area = {:.2f}'.format(integrate(fraction_of_positives, reference, perf_calibrated)))

    ax1.set_ylabel("Fraction of positives", fontsize=35)
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right",  ncol=2, fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=25)
    # ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value", fontsize=35)
    ax2.set_ylabel("Count", fontsize=40)
    ax2.legend(loc="upper center", ncol=2, fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=25)

    # plt.tight_layout()
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    #mng.window.showMaximized()
    mng.window.state('zoomed')
    plt.show(block=False)
    plt.pause(0.001)
    # manager = plt.get_current_fig_manager()
    # manager.window.showMaximized()
    try:
        fig.savefig(os.path.join(orig_path, 'results', 'reliability_diagram', 'rd_diagram' + file_suffix + '.png'))
        fig.savefig(os.path.join(orig_path, 'results', 'reliability_diagram', 'rd_diagram' + file_suffix + '.eps'))
    except:
        os.makdirs(os.path.join(orig_path, 'results', 'reliability_diagram'))
        fig.savefig(os.path.join(orig_path, 'results', 'reliability_diagram', 'rd_diagram' + file_suffix + '.png'))
        fig.savefig(os.path.join(orig_path, 'results', 'reliability_diagram', 'rd_diagram' + file_suffix + '.eps'))
    plt.close()


def printing_results(df, txt_vector):

    print('-------------------------------------------------------------------------------')
    print('                     {} results'.format(txt_vector))
    print('-------------------------------------------------------------------------------')
    print('             ', *columns, sep='\t')
    print('Acc:         ', *list(df.loc[10, 'Acc']), sep='\t')
    print('AUC:         ', *list(df.loc[10, 'AUC']), sep='\t')
    print('F1:          ', *list(df.loc[10, 'F1']), sep='\t')
    print('Acc_unc:     ', *list(df.loc[10, 'Acc_unc']), sep='\t')
    print('AUC_unc:     ', *list(df.loc[10, 'AUC_unc']), sep='\t')
    print('F1_unc:      ', *list(df.loc[10, 'F1_unc']), sep='\t')
    print('Acc_calprob: ', *list(df.loc[10, 'Acc_calprob']), sep='\t')
    print('AUC_calprob: ', *list(df.loc[10, 'AUC_calprob']), sep='\t')
    print('F1_calprob:  ', *list(df.loc[10, 'F1_calprob']), sep='\t')


def collect_metric4uncertainty(y_true, y_logits, percentage, dictionary, unc_vector, run, subject, thr_low, thr_up, metric, criterion):
    
    thresholds = np.linspace(thr_low, thr_up, 20)
    ind = 0
    
    for thr in thresholds:
        if criterion == 'entropy':
            y_sfx = np.multiply(y_logits, (unc_vector <= thr).reshape((-1, 1)).astype(np.int64))
        else:
            y_sfx = np.multiply(y_logits, (unc_vector >= thr).reshape((-1, 1)).astype(np.int64))
        if metric == 'auc':
            fpr, tpr, _ = roc_curve(y_true.astype('int'), y_sfx, pos_label=1, drop_intermediate=True)
            dictionary[run, ind, subject] = auc(fpr, tpr)
        elif metric == 'f1':
            tmp = np.concatenate((np.zeros((y_sfx.shape[0], 1)), y_sfx.squeeze().reshape(-1, 1)), axis=1)
            pred = np.array(np.argmax(tmp, axis=1))
            dictionary[run, ind, subject] = f1_score(y_true.astype('int').squeeze(), pred, zero_division='warn')
        elif metric == 'precision':
            tmp = np.concatenate((np.zeros((y_sfx.shape[0], 1)), y_sfx.squeeze().reshape(-1, 1)), axis=1)
            pred = np.array(np.argmax(tmp, axis=1)).reshape(-1, 1)
            dictionary[run, ind, subject] = precision_score(y_true, pred, zero_division='warn')

        if criterion == 'entropy':
            percentage[run, ind, subject] = 100 * np.sum(np.asarray(unc_vector <= thr).astype('int')) / \
                                        y_true.shape[0]
        else:
            percentage[run, ind, subject] = 100 * np.sum(np.asarray(unc_vector >= thr).astype('int')) / \
                                        y_true.shape[0]

        ind += 1

    return dictionary


# ---------------------------------------------------------------------------------------
#         Aux. Functions
# ---------------------------------------------------------------------------------------
# Which paper for comparison
Nastaran = True
if Nastaran:
    author = r'nastaran\Boxuan_ESDB_widernet8x_dense128'
    # 0 - Nastaran, 2000 - Sadouk
    training_batch = 0  # value used in the training script
    if 'Boxuan' in author:
        roc_approach = ['ROC WiderNet8x Original', 'ROC Bayesian', 'Comparison WiderNet8x vs Bayesian']
    else:
        roc_approach = ['ROC Rad Original Approach', 'ROC Bayesian Approach', 'Comparison Rad vs Bayesian']
else:
    author = 'sadouk\ESDB'
    # 0 - Nastaran, 2000 - Sadouk
    training_batch = 2000  # value used in the training script
    roc_approach = ['ROC Sadouk Original Approach', 'ROC Bayesian Approach', 'Comparison Sadouk vs Bayesian']

if platform.system() != 'Windows':
    orig_path = r'C:\path\to\src\bayes_approach_sadouk'
    load_model_path = r'RockingMotion/Journal/src'
else:
    load_model_path = r'C:\path\to\src\transferlearning_ESDB_equipped_3'
    orig_path = r'C:\path\to\src\bayes_approach_'+ author

if 'ESDB' not in author:
    session = 'Subj.:'
    total_subjs = 6
else:
    session = 'Sess.:'
    total_subjs = 7

list_of_subjs = [i for i in range(total_subjs)]
bayes_conditions = [True]
total_runs = [i for i in range(10)]
mean_fpr = np.linspace(0, 1, 3500)

# Dataframe to hold end results
columns = ['s' + str(i) for i in list_of_subjs]
rows = []
for i in ['Acc', 'F1', 'AUC']:
    for j in ['', '_unc', '_calprob']:
        rows.extend([i + j])
runs = total_runs + [len(total_runs)]
indexer = pd.MultiIndex.from_product([rows, columns])
df_results = pd.DataFrame(columns=indexer, index=runs)

# For faster execution
skip = True

mean_tpr = [0]*2
mean_auc = [0]*2
std_auc = [0]*2
std_tpr = [0]*2
tprs_upper = [0]*2
tprs_lower = [0]*2
# ---- Uncertainty related
mean_tpr_unc = [0]*2
mean_auc_unc = [0]*2
std_auc_unc = [0]*2
std_tpr_unc = [0]*2
tprs_upper_unc = [0]*2
tprs_lower_unc = [0]*2
precision = [0]*2

if os.path.exists(os.path.join(orig_path, 'results')):
    print('Folder is ready!')
else:
    os.makedirs(os.path.join(orig_path, 'results'))

# Name of approach index
ind = 0
# For saving file name purposes only
unc_thr = 0
cal_prob_thr = 0
for cond in bayes_conditions:

    fig, ax = plt.subplots()
    approach_dict = {
        'aggregate_tpr': np.zeros((len(total_runs), 3500, len(list_of_subjs))),
        'aggregate_auc': np.zeros((len(total_runs), len(list_of_subjs))),
        'aggregate_tn': np.zeros((len(total_runs), 100, len(list_of_subjs))),
        'aggregate_tp': np.zeros((len(total_runs), 100, len(list_of_subjs))),
        'aggregate_fn': np.zeros((len(total_runs), 100, len(list_of_subjs))),
        'aggregate_fp': np.zeros((len(total_runs), 100, len(list_of_subjs))),
        'aggregate_mcc': np.zeros((len(total_runs), 100, len(list_of_subjs))),
        'aggregate_ck': np.zeros((len(total_runs), 100, len(list_of_subjs))),
        'aggregate_f1': np.zeros((len(total_runs), 100, len(list_of_subjs))),
        'aggregate_perc': np.zeros((len(total_runs), 100, len(list_of_subjs))),
        'tprs': np.zeros((len(total_runs), 100, len(list_of_subjs))),
        'specificity': np.zeros((len(total_runs), 100, len(list_of_subjs))),
        'sensitivity': np.zeros((len(total_runs), 100, len(list_of_subjs))),
        'precision': np.zeros((len(total_runs), 100, len(list_of_subjs))),
        'aggregate_perctg_unc': np.zeros((len(total_runs), 20, len(list_of_subjs))),
        'aggregate_perctg_calprob': np.zeros((len(total_runs), 20, len(list_of_subjs))),
        'AUC_unc': np.zeros((len(total_runs), 20, len(list_of_subjs))),
        'AUC_calprob': np.zeros((len(total_runs), 20, len(list_of_subjs))),
        'precision_unc': np.zeros((len(total_runs), 20, len(list_of_subjs))),
        'precision_calprob': np.zeros((len(total_runs), 20, len(list_of_subjs))),
        'f1_unc': np.zeros((len(total_runs), 20, len(list_of_subjs))),
        'f1_calprob': np.zeros((len(total_runs), 20, len(list_of_subjs)))
    }
    if cond:
        fig_unc, ax_unc = plt.subplots()
        fig_calprob, ax_calprob = plt.subplots()
        # Uncertainty related arrays
        unc_dict = copy.deepcopy(approach_dict)
        calprob_dict = copy.deepcopy(approach_dict)
        agg_y_sfx_sub_all = []
        agg_y_sfx_unc_sub_all = []
        agg_y_sfx_calprob_sub_all = []
        agg_y_true_sub_all = []
        agg_y_true_unc_sub_all = []
        agg_y_true_calprob_sub_all = []
        percent_pts_unc = []
        percent_pts_calprob = []
    # Starter subject indexer
    indexer = 0


    for subj in list_of_subjs:
        # agg_unc_sub_corr = 0
        # agg_unc_sub_incorr = 0
        # agg_calprob_sub_corr = 0
        # agg_calprob_sub_incorr = 0
        if cond:
            percent_pts_unc.append(0)
            percent_pts_calprob.append(0)

        for exec_run in total_runs:
            filename = 'data_subj' + str(subj) + '_out_baye' + str(cond) + '_' + str(exec_run) + '.mat'
            filepath = os.path.join(orig_path, filename)
            var = sio.loadmat(filepath)
            y_pred = var['test_pred_svm']
            y_true = var['test_gt'][training_batch:]
            y_sfx = var['svm_softmax'][:, 1].reshape((-1, 1))

            correct = np.bitwise_and(y_true > 0, y_pred > 0)
            incorrect = np.bitwise_and(np.bitwise_not(y_true > 0), y_pred > 0)

            # approach_dict = compute_over_runs(approach_dict, y_true, y_sfx, indexer)

            if cond:
                # y_sfx = var['test_cali_prob_all'][training_batch:, :].reshape((-1, 1))
                test_unc = var['test_uncertainties_all'][training_batch:, :]
                cal_prob = var['test_cali_prob_all'][training_batch:, :]
                unc_thr = 0.4
                cal_prob_thr = 0.85
                # Use the following two lines if wanted to check setting sample out of criterion to zero
                y_sfx_unc = np.multiply(y_sfx, np.asarray((test_unc[:, 1] <= unc_thr).reshape((-1, 1))))
                y_pred_unc = np.multiply(y_pred, np.asarray((test_unc[:, 1] <= unc_thr).reshape((-1, 1))).astype('int'))
                # Use the following three lines if wanted to check scenario excluding samples out of the criterion
                # y_sfx_unc = y_sfx[test_unc[training_batch:, 2] <= unc_thr]
                # y_true_unc = y_true[test_unc[training_batch:, 2] <= unc_thr]
                # y_pred_unc = y_pred[test_unc[training_batch:, 2] <= unc_thr]
                # Use the following two lines if wanted to check setting sample out of criterion to zero
                y_sfx_calprob = np.multiply(y_sfx, np.asarray((cal_prob.ravel() >= cal_prob_thr).reshape((-1, 1))))
                y_pred_calprob = np.multiply(y_pred, np.asarray((cal_prob.ravel() >= cal_prob_thr).reshape((-1, 1))).astype('int'))
                # Use the following three lines if wanted to check scenario excluding samples out of the criterion
                # y_sfx_calprob = y_sfx[cal_prob[training_batch:].ravel() >= cal_prob_thr]
                # y_true_calprob = y_true[cal_prob[training_batch:].ravel() >= cal_prob_thr]
                # y_pred_calprob = y_pred[cal_prob[training_batch:].ravel() >= cal_prob_thr]
                print('Subj {}, Run {}'.format(subj, exec_run))
                percent_pts_unc[indexer] += (100 * np.sum(np.asarray(test_unc[:, 1] <= unc_thr).astype('int'))/y_sfx.shape[0])/len(total_runs)
                percent_pts_calprob[indexer] += (100 * np.sum(np.asarray(cal_prob.ravel() >= cal_prob_thr).astype('int')) / y_sfx.shape[0])/len(total_runs)
                ## Use the following two lines if wanted to check setting sample out of criterion to zero
                ## unc_dict = compute_over_runs(unc_dict, y_true, y_sfx_unc, indexer)
                ## calprob_dict = compute_over_runs(calprob_dict, y_true, y_sfx_calprob, indexer)
                # Use y_true_unc and y_true_calprob to check scenario excluding samples out of the criterion
                # unc_dict = compute_over_runs(unc_dict, y_true_unc, y_sfx_unc, indexer)
                # calprob_dict = compute_over_runs(calprob_dict, y_true_calprob, y_sfx_calprob, indexer)


                # ------------------------------------
                # Gather AUC for several uncertainty thresholds
                # ------------------------------------
                approach_dict['AUC_unc'] = collect_metric4uncertainty(y_true, y_sfx, approach_dict['aggregate_perctg_unc'],
                                                       approach_dict['AUC_unc'], test_unc[:, 1], exec_run, subj,
                                                       np.amin(test_unc[:, 1]), np.amax(test_unc[:, 1]), 'auc', 'entropy')
                approach_dict['AUC_calprob'] = collect_metric4uncertainty(y_true, y_sfx, approach_dict['aggregate_perctg_calprob'],
                                                           approach_dict['AUC_calprob'], cal_prob.ravel(), exec_run,
                                                           subj, 0.6, 1, 'auc', 'calprob')
                # # ------------------------------------
                # # Gather F1 for several uncertainty thresholds
                # # ------------------------------------
                # approach_dict['f1_unc'] = collect_metric4uncertainty(y_true, y_sfx, approach_dict['aggregate_perctg_unc'],
                #                                        approach_dict['f1_unc'], test_unc[:, 1], exec_run, subj,
                #                                        np.amin(test_unc[:, 1]), np.amax(test_unc[:, 1]), 'f1')
                # approach_dict['f1_calprob'] = collect_metric4uncertainty(y_true, y_sfx, approach_dict['aggregate_perctg_calprob'],
                #                                            approach_dict['f1_calprob'], cal_prob.ravel(), exec_run,
                #                                            subj, 0, 1, 'f1')
                # # ------------------------------------
                # # Gather Precision for several uncertainty thresholds
                # # ------------------------------------
                # approach_dict['precision_unc'] = collect_metric4uncertainty(y_true, y_sfx, approach_dict['aggregate_perctg_unc'],
                #                                        approach_dict['precision_unc'], test_unc[:, 1], exec_run, subj,
                #                                        np.amin(test_unc[:, 1]), np.amax(test_unc[:, 1]), 'precision')
                # approach_dict['precision_calprob'] = collect_metric4uncertainty(y_true, y_sfx, approach_dict['aggregate_perctg_calprob'],
                #                                            approach_dict['precision_calprob'], cal_prob.ravel(), exec_run,
                #                                            subj, 0, 1, 'precision')
                # ------------------------------------
                # Aggregations for uncertainty related histograms and reliability diagrams
                # ------------------------------------
                try:
                    agg_unc_sub_corr = np.concatenate((agg_unc_sub_corr, test_unc[correct.reshape(-1), :]), axis=0)
                    agg_unc_sub_incorr = np.concatenate((agg_unc_sub_incorr, test_unc[incorrect.reshape(-1), :]), axis=0)
                    agg_calprob_sub_corr = np.concatenate((agg_calprob_sub_corr, cal_prob[correct[:, 0]]), axis=0)
                    agg_calprob_sub_incorr = np.concatenate((agg_calprob_sub_incorr, cal_prob[incorrect[:, 0]]), axis=0)
                    agg_y_sfx_sub = np.concatenate((agg_y_sfx_sub, y_sfx), axis=0)
                    agg_y_sfx_unc_sub = np.concatenate((agg_y_sfx_unc_sub, y_sfx_unc), axis=0)
                    agg_y_sfx_calprob_sub = np.concatenate((agg_y_sfx_calprob_sub, y_sfx_calprob), axis=0)
                    agg_y_true_sub = np.concatenate((agg_y_true_sub, y_true), axis=0)
                    agg_y_true_unc_sub = np.concatenate((agg_y_true_unc_sub, y_true), axis=0)
                    agg_y_true_calprob_sub = np.concatenate((agg_y_true_calprob_sub, y_true), axis=0)
                except:
                    print('Creating copies of uncertainty for subj {}, run {}'.format(subj, exec_run))
                    agg_unc_sub_corr = copy.deepcopy(test_unc[correct.reshape(-1), :])
                    agg_unc_sub_incorr = copy.deepcopy(test_unc[incorrect.reshape(-1), :])
                    agg_calprob_sub_corr = copy.deepcopy(cal_prob[correct.reshape(-1)])
                    agg_calprob_sub_incorr = copy.deepcopy(cal_prob[incorrect.reshape(-1)])
                    agg_y_sfx_sub = copy.deepcopy(y_sfx)
                    agg_y_sfx_unc_sub = copy.deepcopy(y_sfx_unc)
                    agg_y_sfx_calprob_sub = copy.deepcopy(y_sfx_calprob)
                    agg_y_true_sub = copy.deepcopy(y_true)
                    agg_y_true_unc_sub = copy.deepcopy(y_true)
                    agg_y_true_calprob_sub = copy.deepcopy(y_true)

                # ------------------------------------
                # Record end results per run
                # ------------------------------------
                # Use the following two lines if wanted to check setting sample out of criterion to zero
                df_results.loc[exec_run, ('Acc_unc', 's' + str(subj))] = np.mean(y_true == y_pred_unc)
                df_results.loc[exec_run, ('F1_unc', 's' + str(subj))] = f1_score(y_true, y_pred_unc, zero_division='warn')
                # Use y_true_unc and y_true_calprob to check scenario excluding samples out of the criterion
                # df_results.loc[exec_run, ('Acc_unc', 's' + str(subj))] = np.mean(y_true_unc == y_pred_unc)
                # df_results.loc[exec_run, ('F1_unc', 's' + str(subj))] = f1_score(y_true_unc, y_pred_unc, zero_division='warn')
                try:
                    # Use y_true_unc and y_true_calprob to check scenario excluding samples out of the criterion
                    # fpr, tpr, _ = roc_curve(y_true_unc, y_sfx_unc, pos_label=1, drop_intermediate=True)
                    # Use the following line if wanted to check setting sample out of criterion to zero
                    fpr, tpr, _ = roc_curve(y_true, y_sfx_unc, pos_label=1, drop_intermediate=True)
                    df_results.loc[exec_run, ('AUC_unc', 's' + str(subj))] = auc(fpr, tpr)
                except:
                    print('Not possible to calculate AUC for Subject {}, Run {}'.format(subj, exec_run))
                    df_results.loc[exec_run, ('AUC_unc', 's' + str(subj))] = 0

                # Use y_true_unc and y_true_calprob to check scenario excluding samples out of the criterion
                # df_results.loc[exec_run, ('Acc_calprob', 's' + str(subj))] = np.mean(y_true_calprob == y_pred_calprob)
                # df_results.loc[exec_run, ('F1_calprob', 's' + str(subj))] = f1_score(y_true_calprob, y_pred_calprob, zero_division='warn')
                # Use the following two lines if wanted to check setting sample out of criterion to zero
                df_results.loc[exec_run, ('Acc_calprob', 's' + str(subj))] = np.mean(y_true == y_pred_calprob)
                df_results.loc[exec_run, ('F1_calprob', 's' + str(subj))] = f1_score(y_true, y_pred_calprob, zero_division='warn')
                try:
                    # Use y_true_unc and y_true_calprob to check scenario excluding samples out of the criterion
                    # fpr, tpr, _ = roc_curve(y_true_calprob, y_sfx_calprob, pos_label=1, drop_intermediate=True)
                    # Use the following line if wanted to check setting sample out of criterion to zero
                    fpr, tpr, _ = roc_curve(y_true, y_sfx_calprob, pos_label=1, drop_intermediate=True)
                    df_results.loc[exec_run, ('AUC_calprob', 's' + str(subj))] = auc(fpr, tpr)
                except:
                    print('Not possible to calculate AUC for Subject {}, Run {}'.format(subj, exec_run))
                    df_results.loc[exec_run, ('AUC_calprob', 's' + str(subj))] = 0

            df_results.loc[exec_run, ('Acc', 's' + str(subj))] = np.mean(y_true == y_pred)
            df_results.loc[exec_run, ('F1', 's' + str(subj))] = f1_score(y_true, y_pred, zero_division='warn')
            fpr, tpr, _ = roc_curve(y_true, y_sfx, pos_label=1, drop_intermediate=True)
            df_results.loc[exec_run, ('AUC', 's' + str(subj))] = auc(fpr, tpr)

        # Summarize values
        # Record end results per run
        df_results.loc[exec_run + 1, ('Acc_unc', 's' + str(subj))] = df_results['Acc_unc'].loc[:exec_run, 's' + str(subj)].mean()
        df_results.loc[exec_run + 1, ('F1_unc', 's' + str(subj))] = df_results['F1_unc'].loc[:exec_run, 's' + str(subj)].mean()
        df_results.loc[exec_run + 1, ('AUC_unc', 's' + str(subj))] = df_results['AUC_unc'].loc[:exec_run, 's' + str(subj)].mean()
        df_results.loc[exec_run + 1, ('Acc_calprob', 's' + str(subj))] = df_results['Acc_calprob'].loc[:exec_run, 's' + str(subj)].mean()
        df_results.loc[exec_run + 1, ('F1_calprob', 's' + str(subj))] = df_results['F1_calprob'].loc[:exec_run, 's' + str(subj)].mean()
        df_results.loc[exec_run + 1, ('AUC_calprob', 's' + str(subj))] = df_results['AUC_calprob'].loc[:exec_run, 's' + str(subj)].mean()
        df_results.loc[exec_run + 1, ('Acc', 's' + str(subj))] = df_results['Acc'].loc[:exec_run, 's' + str(subj)].mean()
        df_results.loc[exec_run + 1, ('F1', 's' + str(subj))] = df_results['F1'].loc[:exec_run, 's' + str(subj)].mean()
        df_results.loc[exec_run + 1, ('AUC', 's' + str(subj))] = df_results['AUC'].loc[:exec_run, 's' + str(subj)].mean()

        # ------------------------------------
        # iterate over axis 0 only
        # ------------------------------------
        axis_method = 0
        if cond:
            # ---- Uncertainty related
            app_method = 'unc'
            # output_mean_metrics, output_metric_names = generate_twin_plots_for_each_subject(unc_dict,
            #                                                                                 axis_method, indexer,
            #                                                                                 app_method)
            # Generate plots for all metrics per subject
            graph_axis = ['Threshold', '', 'metrics_' + str(subj) + '_unc']
            # iterate_and_plot(output_mean_metrics, output_metric_names, graph_axis)
            # ---- Calibrated probability related
            app_method = 'calprob'
            # output_mean_metrics, output_metric_names = generate_twin_plots_for_each_subject(calprob_dict,
            #                                                                                 axis_method, indexer,
            #                                                                                 app_method)
            # Generate plots for all metrics per subject
            graph_axis = ['Threshold', '', 'metrics_' + str(subj) + '_calprob']
            # iterate_and_plot(output_mean_metrics, output_metric_names, graph_axis)

            # labels = ['Entropy', 'MI', 'subj_' + str(subj) + '_correct']
            # plot_histograms(agg_unc_sub_corr[:, 1:], labels)
            # labels = ['Entropy', 'MI', 'subj_' + str(subj) + '_incorrect']
            # plot_histograms(agg_unc_sub_incorr[:, 1:], labels)
            try:
                agg_unc_sub_corr_all = np.concatenate((agg_unc_sub_corr_all, agg_unc_sub_corr), axis=0)
                agg_unc_sub_incorr_all = np.concatenate((agg_unc_sub_incorr_all, agg_unc_sub_incorr), axis=0)
                agg_calprob_sub_corr_all = np.concatenate((agg_calprob_sub_corr_all, agg_calprob_sub_corr), axis=0)
                agg_calprob_sub_incorr_all = np.concatenate((agg_calprob_sub_incorr_all, agg_calprob_sub_incorr), axis=0)
            except:
                print('Creating copies of uncertainty for subj {}, run {}'.format(subj, exec_run))
                agg_unc_sub_corr_all = copy.deepcopy(agg_unc_sub_corr)
                agg_unc_sub_incorr_all = copy.deepcopy(agg_unc_sub_incorr)
                agg_calprob_sub_corr_all = copy.deepcopy(agg_calprob_sub_corr)
                agg_calprob_sub_incorr_all = copy.deepcopy(agg_calprob_sub_incorr)

            agg_y_sfx_sub_all.append(agg_y_sfx_sub)
            agg_y_sfx_unc_sub_all.append(agg_y_sfx_unc_sub)
            agg_y_sfx_calprob_sub_all.append(agg_y_sfx_calprob_sub)
            agg_y_true_sub_all.append(agg_y_true_sub)
            agg_y_true_unc_sub_all.append(agg_y_true_unc_sub)
            agg_y_true_calprob_sub_all.append(agg_y_true_calprob_sub)

        app_method = 'baye' + str(cond)
        # output_mean_metrics, output_metric_names = generate_twin_plots_for_each_subject(approach_dict,
        #                                                                                 axis_method, indexer, app_method)
        # ------------------------------------
        # Generate plots for all metrics per subject
        # ------------------------------------
        graph_axis = ['Threshold', ' ', roc_approach[ind][4:10] + '_' + str(cond) + '_metrics_all']
        # iterate_and_plot(output_mean_metrics, output_metric_names, graph_axis)

        # ROC-AUC per subject
        y_subj = np.mean(approach_dict['aggregate_tpr'][:, :, indexer], axis=0)
        ax.plot(mean_fpr, y_subj, label=session + ' {} (AUC: {:.2f}, prec.: {:.2f})'.format(
            subj, np.mean(approach_dict['aggregate_auc'][:, indexer], axis=0),
            np.nanmean(approach_dict['precision'][:, :, indexer], axis=(0, 1))))

        if cond:
            # ---- Uncertainty related
            y_subj = np.mean(unc_dict['aggregate_tpr'][:, :, indexer], axis=0)
            ax_unc.plot(mean_fpr, y_subj, label=session + ' {} (AUC: {:.2f}, Selected by       unc. <= {:.2f}, prec.: {:.2f}, % pts.: {:.1f})'.format(
                subj, np.mean(unc_dict['aggregate_auc'][:, indexer], axis=0), unc_thr,
                np.nanmean(unc_dict['precision'][:, :, indexer], axis=(0, 1)), percent_pts_unc[indexer]))
            # # ---- Calibrated probability related
            y_subj = np.mean(calprob_dict['aggregate_tpr'][:, :, indexer], axis=0)
            ax_calprob.plot(mean_fpr, y_subj, label=session + ' {} (AUC: {:.2f}, Selected by cal.prob.>= {:.2f}, prec.: {:.2f}, % pts.: {:.1f})'.format(
                subj, np.mean(calprob_dict['aggregate_auc'][:, indexer], axis=0), cal_prob_thr,
                np.nanmean(calprob_dict['precision'][:, :, indexer], axis=(0, 1)), percent_pts_calprob[indexer]))
            #

        indexer += 1

    # ------------------------------------
    # Generate plots metrics over all subjects
    # ------------------------------------
    axis_method = (0, 2)
    # output_mean_metrics, output_metric_names = generate_twin_plot_over_all_subjects(approach_dict, axis_method, roc_approach[ind][4:10], roc_approach[ind])
    graph_axis = ['Threshold', ' ', roc_approach[ind][4:10] + '_metrics_' + str(subj)]
    # iterate_and_plot(output_mean_metrics, output_metric_names, graph_axis)
    if cond:
        # # ---- Uncertainty related
        # generate_twin_plot_over_all_subjects(unc_dict, axis_method, roc_approach[ind][4:10], '_unc')
        # # ---- Calibrated probability related
        # generate_twin_plot_over_all_subjects(calprob_dict, axis_method, roc_approach[ind][4:10], '_calprob')

        ax_unc.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gainsboro',
                label='Chance', alpha=.8)
        ax_calprob.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gainsboro',
                label='Chance', alpha=.8)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gainsboro',
            label='Chance', alpha=.8)

    # ------------------------------------
    # Average ROC over all subjects
    # ------------------------------------
    mean_tpr[ind] = np.mean(approach_dict['aggregate_tpr'], axis=(0, 2))
    mean_tpr[ind][-1] = 1.0
    mean_auc[ind] = auc(mean_fpr, mean_tpr[ind])
    std_auc[ind] = np.std(approach_dict['aggregate_auc'], axis=(0, 1))
    ax.plot(mean_fpr, mean_tpr[ind], color='b',
            label=r'Mean ROC (AUC: {:0.2f} $\pm$ {:0.2f}, mean prec.: {:.2f})'.format(
                mean_auc[ind], std_auc[ind], np.nanmean(approach_dict['precision'])),
            lw=2, alpha=.8)

    # Fill graph with std
    std_tpr[ind] = np.std(approach_dict['aggregate_tpr'], axis=(0, 2))
    tprs_upper[ind] = np.minimum(mean_tpr[ind] + std_tpr[ind], 1)
    tprs_lower[ind] = np.maximum(mean_tpr[ind] - std_tpr[ind], 0)
    ax.fill_between(mean_fpr, tprs_lower[ind], tprs_upper[ind], color='orange', alpha=.2,
                    label=r'$\pm$ 1 std. dev. for ' + roc_approach[ind][4:8])
    if cond:
        # ---- Uncertainty related
        mean_tpr_unc[0] = np.mean(unc_dict['aggregate_tpr'], axis=(0, 2))
        mean_tpr_unc[0][-1] = 1.0
        mean_auc_unc[0] = auc(mean_fpr, mean_tpr_unc[0])
        std_auc_unc[0] = np.std(unc_dict['aggregate_auc'], axis=(0, 1))
        ax_unc.plot(mean_fpr, mean_tpr_unc[0], color='mediumorchid',
                label=r'Mean ROC (AUC: {:0.2f} $\pm$ {:0.2f}, Selected by       unc. <= {:.1f}, prec.: {:.2f}, % pts.: {:.1f})'.format(
                    mean_auc_unc[0], std_auc_unc[0], unc_thr, np.nanmean(unc_dict['precision']), sum(percent_pts_unc)/len(percent_pts_unc)),
                lw=2, alpha=.8)

        # Fill graph with std
        std_tpr_unc[0] = np.std(unc_dict['aggregate_tpr'], axis=(0, 2))
        tprs_upper_unc[0] = np.minimum(mean_tpr_unc[0] + std_tpr_unc[0], 1)
        tprs_lower_unc[0] = np.maximum(mean_tpr_unc[0] - std_tpr_unc[0], 0)
        ax_unc.fill_between(mean_fpr, tprs_lower_unc[0], tprs_upper_unc[0], color='magenta', alpha=.2,
                        label=r'$\pm$ 1 std. dev., selected by uncertainty')
        # ---- Calibrated probability related
        mean_tpr_unc[1] = np.mean(calprob_dict['aggregate_tpr'], axis=(0, 2))
        mean_tpr_unc[1][-1] = 1.0
        mean_auc_unc[1] = auc(mean_fpr, mean_tpr_unc[1])
        std_auc_unc[1] = np.std(calprob_dict['aggregate_auc'], axis=(0, 1))
        ax_calprob.plot(mean_fpr, mean_tpr_unc[1], color='darkseagreen',
                label=r'Mean ROC (AUC: {:0.2f} $\pm$ {:0.2f}, Selected by cal.prob.>= {:.1f}, prec.: {:.2f}, % pts.: {:.1f})'.format(
                    mean_auc_unc[1], std_auc_unc[1], cal_prob_thr,
                    np.nanmean(calprob_dict['precision']), sum(percent_pts_calprob)/len(percent_pts_calprob)), lw=2, alpha=.8)

        # Fill graph with std
        std_tpr_unc[1] = np.std(calprob_dict['aggregate_tpr'], axis=(0, 2))
        tprs_upper_unc[1] = np.minimum(mean_tpr_unc[1] + std_tpr_unc[1], 1)
        tprs_lower_unc[1] = np.maximum(mean_tpr_unc[1] - std_tpr_unc[1], 0)
        ax_calprob.fill_between(mean_fpr, tprs_lower_unc[1], tprs_upper_unc[1], color='lime', alpha=.2,
                        label=r'$\pm$ 1 std. dev., selected by cal. prob.')
        precision_unc = [np.nanmean(unc_dict['precision']), np.nanmean(calprob_dict['precision'])]

        # ------------------------------------
        # Histograms for uncertainty
        # ------------------------------------
        # labels = ['Entropy', 'MI', 'all_subjs_correct']
        # plot_histograms(agg_unc_sub_corr_all[:, 1:], labels)
        # labels = ['Entropy', 'MI', 'all_subjs_incorrect']
        # plot_histograms(agg_unc_sub_incorr_all[:, 1:], labels)

        # ------------------------------------
        # Scatterplots for uncertainty
        # ------------------------------------
        # figS = plt
        # sio.savemat(os.path.join(orig_path, 'results', 'uncertainties_ent' + str(unc_thr) + '.mat'),
        #             {'agg_unc_sub_corr_all': agg_unc_sub_corr_all, 'agg_unc_sub_incorr_all': agg_unc_sub_incorr_all})
        # ------------------------------------
        # Reliability Diagram for Uncertainty
        # ------------------------------------
        # reliability_diagram(agg_y_sfx_sub_all, agg_y_true_sub_all, [session + ' ' + str(i) for i in list_of_subjs], '')
        # reliability_diagram(agg_y_sfx_unc_sub_all, agg_y_true_unc_sub_all, ['subj. ' + str(i) for i in list_of_subjs], '_unc')
        # reliability_diagram(agg_y_sfx_calprob_sub_all, agg_y_true_calprob_sub_all, ['subj. ' + str(i) for i in list_of_subjs], '_calprob')

        ax_unc.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
        ax_unc.legend(loc="lower right", fontsize=15)
        ax_unc.set_xlabel('False Positive Rate', fontsize=40)
        ax_unc.set_ylabel('True Positive Rate', fontsize=40)
        ax_unc.tick_params(axis='both', which='major', labelsize=25)
        plt.figure(2)
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        # mng.window.showMaximized()
        # mng.window.state('zoomed')
        plt.show(block=False)
        plt.pause(0.001)
        # fig_unc.savefig(os.path.join(orig_path, 'results', roc_approach[ind] + '_unc.png'))
        # fig_unc.savefig(os.path.join(orig_path, 'results', roc_approach[ind] + '_unc.eps'))
        plt.close(fig_unc)

        ax_calprob.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
        ax_calprob.legend(loc="lower right", fontsize=15)
        ax_calprob.set_xlabel('False Positive Rate', fontsize=40)
        ax_calprob.set_ylabel('True Positive Rate', fontsize=40)
        ax_calprob.tick_params(axis='both', which='major', labelsize=25)
        plt.figure(3)
        mng_calprob = plt.get_current_fig_manager()
        mng_calprob.full_screen_toggle()
        # mng.window.showMaximized()
        # mng.window.state('zoomed')
        plt.show(block=False)
        plt.pause(0.001)
        # fig_calprob.savefig(os.path.join(orig_path, 'results', roc_approach[ind] + '_calprob.png'))
        # fig_calprob.savefig(os.path.join(orig_path, 'results', roc_approach[ind] + '_calprob.eps'))
        plt.close(fig_calprob)

        # -------------------------
        # AUC by uncertainties
        # -------------------------

        fig_auc, ax_auc = plt.subplots()

        xall_unc = np.nanmean(approach_dict['aggregate_perctg_unc'], axis=(0, 2))
        yall_unc = np.nanmean(approach_dict['AUC_unc'], axis=(0, 2))
        # yall_prec_unc = np.nanmean(approach_dict['precision_unc'], axis=(0, 2))
        # yall_f1_unc = np.nanmean(approach_dict['f1_unc'], axis=(0, 2))
        txtall_unc = ['{:.2f}'.format(i) for i in np.linspace(np.amin(test_unc[:, 1]), np.amax(test_unc[:, 1]), 20)]

        xall_calprob = np.nanmean(approach_dict['aggregate_perctg_calprob'], axis=(0, 2))
        yall_calprob = np.nanmean(approach_dict['AUC_calprob'], axis=(0, 2))
        # yall_prec_calprob = np.nanmean(approach_dict['precision_calprob'], axis=(0, 2))
        # yall_f1_calprob = np.nanmean(approach_dict['f1_calprob'], axis=(0, 2))
        txtall_calprob = ['{:.2f}'.format(i) for i in np.linspace(0.6, 1, 20)]

        min_perc = np.amin(np.minimum(xall_unc, xall_calprob))
        max_perc = np.amax(np.maximum(xall_unc, xall_calprob))

        ax_auc.plot(xall_unc, yall_unc, label='AUC - Entropy', marker='P', color='darkgrey')
        ax_auc.plot(xall_calprob, yall_calprob, label='AUC - Cal. prob.', marker='o', color='firebrick')
        # ax_auc.plot(xall_calprob, yall_prec_unc, label='Precision', marker='$P$')
        # ax_auc.plot(xall_calprob, yall_f1_unc, label='F1 score', marker='$F1$')
        fl = 0
        for x,y,t in zip(xall_unc[:-1], yall_unc[:-1], txtall_unc[:-1]):
            ax_auc.text(x, y+fl, t, fontsize=20)
            fl +=0.02
        fl = 0
        for x,y,t in zip(xall_calprob[:-1], yall_calprob[:-1], txtall_calprob[:-1]):
            ax_auc.text(x, y-fl, t, fontsize=20)
            fl += 0.005

        ax_auc.set_xlim([-5, 105])
        ax_auc.legend(fontsize=30)
        ax_auc.set_xlabel('Percentage of samples kept as is', fontsize=40)
        ax_auc.set_ylabel('Score', fontsize=40)
        ax_auc.tick_params(axis='both', which='major', labelsize=25)
        plt.figure(fig_auc.number)
        mng_auc = plt.get_current_fig_manager()
        mng_auc.full_screen_toggle()
        plt.show(block=False)
        plt.pause(0.001)
        fig_auc.savefig(os.path.join(orig_path, 'results', 'AUCxUnc.png'))
        fig_auc.savefig(os.path.join(orig_path, 'results', 'AUCxUnc.eps'))

        with open(os.path.join(orig_path, 'results', 'aucxperc_' + roc_approach[0][4:] + '.pkl'), 'wb') as filehandle:
            pickle.dump([xall_unc, yall_unc, txtall_unc, xall_calprob, yall_calprob, txtall_calprob, min_perc, max_perc],
                        filehandle)

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    ax.legend(loc="lower right", fontsize=20)
    ax.set_xlabel('False Positive Rate', fontsize=40)
    ax.set_ylabel('True Positive Rate', fontsize=40)
    ax.tick_params(axis='both', which='major', labelsize=25)
    plt.figure(fig.number)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    #mng.window.showMaximized()
    #mng.window.state('zoomed')
    plt.show(block=False)
    plt.pause(0.001)
    # fig.savefig(os.path.join(orig_path, 'results', roc_approach[ind] + '.png'))
    # fig.savefig(os.path.join(orig_path, 'results', roc_approach[ind] + '.eps'))
    plt.figure(fig.number)
    precision[ind] = np.nanmean(approach_dict['precision'])
    # df_results.to_csv(os.path.join(orig_path, 'results', 'end_results_' + roc_approach[ind][4:8] + '_ent' + '{:.2f}'.format(unc_thr) + 'cal' + '{:.2f}'.format(cal_prob_thr) + '.csv'), index=True,
    #                   header=True)
    printing_results(df_results, roc_approach[ind][4:8])
    original_stdout = sys.stdout
    # with open(os.path.join(orig_path, 'results', 'end_results_log_' + roc_approach[ind][4:8] + '_ent' + '{:.2f}'.format(unc_thr) + 'cal' + '{:.2f}'.format(cal_prob_thr)+ '.txt'), 'w') as f:
    #     sys.stdout = f  # Change the standard output to the file we created.
    #     printing_results(df_results, roc_approach[ind][4:8])
    #     sys.stdout = original_stdout
    ind += 1
    plt.close(fig)
    # Clean entire datframe
    for col in df_results.columns:
        df_results[col].values[:] = 0

colors_for_plot = ['grey', 'firebrick']
colors_for_std = ['lightgrey', 'lightcoral']
colors_for_plot_unc = ['mediumorchid', 'darkseagreen']
colors_for_std_unc = ['magenta', 'lime']
auc_unc_messages = [' AUC = {:0.2f} $\pm$ {:0.2f}, for unc., prec.: {:.2f}, %pts.:{:.1f}',
                    ' AUC = {:0.2f} $\pm$ {:0.2f}, for cal.prob, prec.: {:.2f}, %pts.:{:.1f}']

percent_pts_unc = [sum(percent_pts_unc)/len(percent_pts_unc), sum(percent_pts_calprob)/len(percent_pts_calprob)]

# with open(os.path.join(orig_path, 'results', 'comparison_final_data_ent' + '{:.2f}'.format(unc_thr) + 'cal' + '{:.2f}'.format(cal_prob_thr) + '.pkl'),
#           'wb') as filehandle:
#     pickle.dump([mean_fpr, mean_tpr, colors_for_plot, roc_approach, mean_auc, std_auc, precision, tprs_lower, tprs_upper,
#                 colors_for_std, mean_tpr_unc, colors_for_plot_unc, auc_unc_messages, mean_auc_unc, std_auc_unc,
#                 precision_unc, percent_pts_unc, tprs_lower_unc, tprs_upper_unc, colors_for_std_unc], filehandle)
#
# fig, ax = plt.subplots()
# for i in range(2):
#
#     ax.plot(mean_fpr, mean_tpr[i], color=colors_for_plot[i],
#                  label=roc_approach[i][4:] + ' (AUC = {:0.2f} $\pm$ {:0.2f}, prec.: {:.2f})'.format(
#                      mean_auc[i], std_auc[i], precision[i]), lw=2, alpha=.8)
#     ax.fill_between(mean_fpr, tprs_lower[i], tprs_upper[i], color=colors_for_std[i], alpha=.2,
#                     label=r'$\pm$ 1 std. dev.')
#     ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=roc_approach[2])
#     ax.legend(loc="lower right")
#     # ---- Uncertainty related
#     ax.plot(mean_fpr, mean_tpr_unc[i], color=colors_for_plot_unc[i],
#                  label='Selected -' + auc_unc_messages[i].format(mean_auc_unc[i], std_auc_unc[i], precision_unc[i],
#                 percent_pts_unc[i]), lw=2, alpha=.8)
#     ax.fill_between(mean_fpr, tprs_lower_unc[i], tprs_upper_unc[i], color=colors_for_std_unc[i], alpha=.2,
#                     label=r'$\pm$ 1 std. dev.')
#     ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=roc_approach[2])
#     ax.legend(loc="lower right")
#
# mng = plt.get_current_fig_manager()
# mng.full_screen_toggle()
# #mng.window.showMaximized()
# #mng.window.state('zoomed')
# plt.show(block=False)
# plt.pause(0.001)
# #manager = plt.get_current_fig_manager()
# #manager.window.showMaximized()
# fig.savefig(os.path.join(orig_path, 'results', roc_approach[2] + '.png'))
# fig.savefig(os.path.join(orig_path, 'results', roc_approach[2] + '.eps'))
# plt.close(fig)
#
#
