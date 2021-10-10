import os
import numpy as np
import platform
import pickle
from matplotlib import pyplot as plt

author0 = r'nastaran\EDAQA_Study1_v3_olddata'
author1 = r'nastaran\EDAQA_Study1_v3_olddata'
author2 = r'nastaran\Boxuan_Study1_wider8x_dense128'
author3 = r'nastaran\Boxuan_Study1_wider8x_dense128'
author4 = r'nastaran\Boxuan_Study2_wider8x_dense128'
author5 = r'nastaran\Boxuan_Study2_wider8x_dense128'
author6 = r'nastaran\Boxuan_Study2_wider16x_dense128'
author7 = r'nastaran\Boxuan_Study2_wider16x_dense128'

orig_path0 = r'C:\Users\rdasilv2\Gdrive\Backup Rafael\Documents\NC State\Research Related\ML and DSP\Proj - Body Rocking\src\bayes_approach_' + author0
orig_path1 = r'C:\Users\rdasilv2\Gdrive\Backup Rafael\Documents\NC State\Research Related\ML and DSP\Proj - Body Rocking\src\bayes_approach_' + author1
orig_path2 = r'C:\Users\rdasilv2\Gdrive\Backup Rafael\Documents\NC State\Research Related\ML and DSP\Proj - Body Rocking\src\bayes_approach_' + author2
orig_path3 = r'C:\Users\rdasilv2\Gdrive\Backup Rafael\Documents\NC State\Research Related\ML and DSP\Proj - Body Rocking\src\bayes_approach_' + author3
orig_path4 = r'C:\Users\rdasilv2\Gdrive\Backup Rafael\Documents\NC State\Research Related\ML and DSP\Proj - Body Rocking\src\bayes_approach_' + author4
orig_path5 = r'C:\Users\rdasilv2\Gdrive\Backup Rafael\Documents\NC State\Research Related\ML and DSP\Proj - Body Rocking\src\bayes_approach_' + author5
orig_path6 = r'C:\Users\rdasilv2\Gdrive\Backup Rafael\Documents\NC State\Research Related\ML and DSP\Proj - Body Rocking\src\bayes_approach_' + author6
orig_path7 = r'C:\Users\rdasilv2\Gdrive\Backup Rafael\Documents\NC State\Research Related\ML and DSP\Proj - Body Rocking\src\bayes_approach_' + author7

# roc_approach = ['ROC Sadouk Original Approach', 'ROC Bayesian Approach', 'Comparison Sadouk vs Bayesian']
# roc_approach0 = ['ROC Rad Original Approach', 'ROC Bayesian Approach', 'Comparison Rad vs Bayesian']
roc_approach0 = ['ROC WiderNet Original Approach', 'ROC Bayesian Approach', 'Comparison WiderNet vs Bayesian']
roc_approach1 = ['ROC WiderNet Original Approach', 'ROC Bayesian Approach', 'Comparison WiderNet vs Bayesian']
roc_approach2 = ['ROC Rad Original Approach', 'ROC Bayesian Approach', 'Comparison Rad vs Bayesian']
roc_approach3 = ['ROC WiderNet Original Approach', 'ROC Bayesian Approach', 'Comparison WiderNet vs Bayesian']
roc_approach4 = ['ROC WiderNet Original Approach', 'ROC Bayesian Approach', 'Comparison WiderNet vs Bayesian']
roc_approach5 = ['ROC Rad Original Approach', 'ROC Bayesian Approach', 'Comparison Rad vs Bayesian']

paths = [orig_path0, orig_path1, orig_path2, orig_path3, orig_path4, orig_path5, orig_path6, orig_path7]
# paths = [orig_path, orig_path1, orig_path2, orig_path2, orig_path2, orig_path2]
# rocs_txts = [roc_approach, roc_approach1, roc_approach2, roc_approach2, roc_approach2, roc_approach2]
# thr_pair = [[0.4, 0.85], [0.1, 0.65], [0.2, 0.75], [0.3, 0.8], [0.4, 0.85], [0.5, 0.9], [0.6, 0.95]]
thr_pair = [[0.4, 0.85], [0.4, 0.85], [0.4, 0.85], [0.4, 0.85]]# [0.4, 0.85], [0.4, 0.85], [0.4, 0.85], [0.4, 0.85]]
list_of_dicts = []

for i, thr in zip(paths, thr_pair):
    with open(os.path.join(i, 'results', 'comparison_final_data_ent' + '{:.2f}'.format(thr[0]) + 'cal' + '{:.2f}'.format(thr[1]) + '.pkl'),
              'rb') as filehandle:
        list_of_vars = pickle.load(filehandle)
        vars = dict()
        vars['mean_fpr'] = list_of_vars[0]
        vars['mean_tpr'] = list_of_vars[1]
        vars['colors_for_plot'] = list_of_vars[2]
        vars['roc_approach'] = list_of_vars[3]
        vars['mean_auc'] = list_of_vars[4]
        vars['std_auc'] = list_of_vars[5]
        vars['precision'] = list_of_vars[6]
        vars['tprs_lower'] = list_of_vars[7]
        vars['tprs_upper'] = list_of_vars[8]
        vars['colors_for_std'] = list_of_vars[9]
        vars['mean_tpr_unc'] = list_of_vars[10]
        vars['colors_for_plot_unc'] = list_of_vars[11]
        vars['auc_unc_messages'] = list_of_vars[12]
        vars['mean_auc_unc'] = list_of_vars[13]
        vars['std_auc_unc'] = list_of_vars[14]
        vars['precision_unc'] = list_of_vars[15]
        vars['percent_pts_unc'] = list_of_vars[16]
        vars['tprs_lower_unc'] = list_of_vars[17]
        vars['tprs_upper_unc'] = list_of_vars[18]
        vars['colors_for_std_unc'] = list_of_vars[19]

        list_of_dicts.append(vars)

use_unc = [False, False, False, False, False, False, False, False]
#0 - run original, 1 - run Bayes, -1, skip original
use_bayes = [0, 1, 0, 1, 0, 1, 0, 1]
# index for selecting which position of array to be displayed from loaded files
# if unc: -1 entropy, 0 calprob
# if not unc: use 0
i = [0, 1, 0, 1, 0, 1, 0, 1]
fig, ax = plt.subplots()
colors_for_plot = ['grey', 'steelblue', 'red', 'green']
colors_for_std = ['lightgrey', 'lightskyblue', 'lightcoral', 'aquamarine']
colors_for_plot_unc = ['mediumorchid', 'darkseagreen']
colors_for_std_unc = ['magenta', 'lime']
# legends = ['Original', 'Cal. Prob.: 0.65', 'Cal. Prob.: 0.75', 'Cal. Prob.: 0.80', 'Cal. Prob.: 0.85', 'Cal. Prob.: 0.90', 'Cal. Prob.: 0.95']
# legends = ['Original', 'Entropy: 0.6', 'Entropy: 0.5', 'Entropy: 0.4', 'Entropy: 0.3', 'Entropy: 0.2', 'Entropy: 0.1']
legends = ['Rad Original', 'Rad - Bayes', 'WiderNet 4x', 'WiderNet 4x - Bayes', 'WiderNet 8x', 'WiderNet 8x - Bayes', 'WiderNet 16x', 'WiderNet 16x - Bayes']
ind = 0

#z['roc_approach'][i][:4]

for z in list_of_dicts:
    if use_bayes[ind] > -1:
        # ax.plot(z['mean_fpr'], z['mean_tpr'][i[ind]], color=colors_for_plot[ind],
        #         label=legends[ind] + ' (AUC = {:0.2f} $\pm$ {:0.2f}, prec.: {:.2f})'.format(
        #                  z['mean_auc'][i[ind]], 1, z['precision'][i[ind]]), lw=2, alpha=.8)
        ax.plot(z['mean_fpr'], z['mean_tpr'][i[ind]],
                label=legends[ind] + ' (AUC = {:0.2f} $\pm$ {:0.2f}, prec.: {:.2f})'.format(
                         z['mean_auc'][i[ind]], z['std_auc'][i[ind]], z['precision'][i[ind]]), lw=2, alpha=.8)
        # ax.fill_between(z['mean_fpr'], z['tprs_lower'][i[ind]], z['tprs_upper'][i[ind]], color=colors_for_std[ind], alpha=.2,
        #                 label=r'$\pm$ 1 std. dev.')
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
        ax.legend(loc="lower right", fontsize=25)

    if use_unc[ind]:
        # ---- Uncertainty related
        # ax.plot(z['mean_fpr'], z['mean_tpr_unc'][(i[ind]+1)/-1], color=colors_for_plot[ind],
        #              label='Selected -' + z['auc_unc_messages'][i[ind]].format(z['mean_auc_unc'][i[ind]], z['std_auc_unc'][i[ind]],
        #                                                                   z['precision_unc'][i[ind]], z['percent_pts_unc'][i[ind]]),
        #              lw=4, alpha=.8)
        ax.plot(z['mean_fpr'], z['mean_tpr_unc'][(i[ind]+1)//-1],
                     label=legends[ind] + ', AUC = {:0.2f} $\pm$ {:0.2f}, prec.: {:0.2f}, %pts.: {:0.1f}'.format(z['mean_auc_unc'][(i[ind]+1)//-1], z['std_auc_unc'][(i[ind]+1)//-1],
                                                                          z['precision_unc'][(i[ind]+1)//-1], z['percent_pts_unc'][(i[ind]+1)//-1]),
                     lw=4, alpha=.8)
        # ax.fill_between(z['mean_fpr'], z['tprs_lower_unc'][(i[ind]+1)/-1], z['tprs_upper_unc'][i[ind]], color=colors_for_std[ind],
        #                 alpha=.2, label=r'$\pm$ 1 std. dev.')
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
        ax.legend(loc="lower right", fontsize=20)

    ind += 1

ax.set_xlabel('False Positive Rate', fontsize=40)
ax.tick_params(axis='both', which='major', labelsize=25)
ax.set_ylabel('True Positive Rate', fontsize=40)
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
#mng.window.showMaximized()
#mng.window.state('zoomed')
plt.show(block=False)
plt.pause(0.001)
#manager = plt.get_current_fig_manager()
#manager.window.showMaximized()
fig.savefig(os.path.join( r'C:\Users\rdasilv2\Gdrive\Backup Rafael\Documents\NC State\Research Related\ML and DSP\Proj - Body Rocking',
                          'Study2all_new.png'))
fig.savefig(os.path.join( r'C:\Users\rdasilv2\Gdrive\Backup Rafael\Documents\NC State\Research Related\ML and DSP\Proj - Body Rocking',
                          'Study2all_new.eps'))
# fig.savefig(os.path.join( r'C:\Users\rdasilv2\Gdrive\Backup Rafael\Documents\NC State\Research Related\ML and DSP\Proj - Body Rocking',
#                           author.split('\\')[-1] + '_' + author1.split('\\')[-1] + '_' + author2.split('\\')[-1] + '.png'))
# fig.savefig(os.path.join( r'C:\Users\rdasilv2\Gdrive\Backup Rafael\Documents\NC State\Research Related\ML and DSP\Proj - Body Rocking',
#                           author.split('\\')[-1] + '_' + author1.split('\\')[-1] + '_' + author2.split('\\')[-1] + '.eps'))
# fig.savefig(os.path.join( r'C:\Users\rdasilv2\Gdrive\Backup Rafael\Documents\NC State\Research Related\ML and DSP\Proj - Body Rocking',
#                           author.split('\\')[-1] + '_' + author1.split('\\')[-1] + '_' + author2.split('\\')[-1] + '_' + author3.split('\\')[-1] + '.png'))
# fig.savefig(os.path.join( r'C:\Users\rdasilv2\Gdrive\Backup Rafael\Documents\NC State\Research Related\ML and DSP\Proj - Body Rocking',
#                           author.split('\\')[-1] + '_' + author1.split('\\')[-1] + '_' + author2.split('\\')[-1] + '_' + author3.split('\\')[-1] + '.eps'))
plt.close(fig)