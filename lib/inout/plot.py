#
# BootMCHawkes
#
# @authors : Carlotta De Pasquale

from lib import Hawkes as hk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from mpi4py import MPI
import os

def plot_process(dataset):
    model.plot_l()
    model.plot_N()
    plt.show()

def plot_estimate(param, dataset, comm):
    logger=logging.getLogger(param["logger"])
    alpha_all = np.zeros(param['n'])
    beta_all = np.zeros(param['n'])
    mu_all = np.zeros(param['n'])
    
    count = np.zeros(param['size'])
    displ = np.zeros(param['size'])
    
    if param['rank']==0:
        c = 0
        r = 0
        for i in range(param['size']):
            count[c] = dataset['n_local']
            displ[c] = dataset['n_local'] * r
            c += 1
            r += 1
        count[param['size'] - 1] = dataset['n_local'] + (param['n'] % param['size'])
        #print(param['rank'], ' count: ', count, ' displ: ', displ)
    
    comm.Gatherv(dataset['alpha'], [alpha_all, count, displ, MPI.DOUBLE], root = 0)
    comm.Gatherv(dataset['beta'], [beta_all, count, displ, MPI.DOUBLE], root = 0)
    comm.Gatherv(dataset['mu'], [mu_all, count, displ, MPI.DOUBLE], root = 0)

    if param['rank'] == 0:
        # est_par = {'alpha': alpha_all, 'beta': beta_all, 'mu': mu_all}
        # df = pd.DataFrame(est_par, columns=['alpha', 'beta', 'mu'])
        max_alpha = np.max(alpha_all)
        min_alpha = np.min(alpha_all)
        min_beta = np.min(beta_all)
        min_mu = np.min(mu_all)
        logger.debug("Alpha max: " + str(max_alpha))
        logger.debug("Alpha, beta, mu min: " + str(min_alpha) + str(min_beta) + str(min_mu))
        
        fig, ax =plt.subplots(1,3)
        plt.subplots_adjust(wspace = 0.35)
        a = sns.histplot(alpha_all, ax=ax[0])
        b = sns.histplot(beta_all, ax=ax[1])
        m = sns.histplot(mu_all, ax=ax[2])
        vlines = [param['alpha'], param['beta'], param['mu']]
        v=0
        for ax in fig.axes:
            ax.axvline(vlines[v], color='red')
            v += 1
        a.set_title('alpha')
        b.set_title('beta')
        m.set_title('mu')
        plt_name = param['dataset_dir'] + param['outprefix'] + "_" + str(param['mu']) + "_" + str(param['alpha']) + "_" + str(param['beta'])+"/"
        plt_name = plt_name + "N_" + str(param['n']) + "_T_" + str(int(param['t'])) + "/" + "allplts.png"
        fig.savefig(plt_name)
        logger.info("Plot saved in " + str(plt_name))
        #aplt = sns.displot(alpha_all, kind='hist')
        # #sns.distplot(alpha_all, hist=False, kde=True, color = 'darkblue', kde_kws={'linewidth': 4})
        # aplt.savefig('alphah.png')
        # plt.clf()
        # bplt = sns.displot(beta_all, kind='hist')
        # #sns.distplot(beta_all, hist=False, kde=True, color = 'g', kde_kws={'linewidth': 4})
        # bplt.savefig('betah.png')
        # plt.clf()
        # mplt = sns.displot(mu_all, kind='hist')
        # #sns.distplot(mu_all, hist=False, kde=True, color = 'm', kde_kws={'linewidth': 4})
        # mplt.savefig("muh.png")

def plot_cint(param, dataset, comm):
    logger=logging.getLogger(param["logger"])

    if param['rank'] == 0:
        error=[]
        stderr_alpha = []
        cint_array = np.array(dataset['cint_alpha_1'])
        i_loc = 0
        j_loc = 0

        if dataset['n_local'] >30:
            for i in  range(30):                  #range(dataset['n_local']):
                error.append([(dataset['alpha'][i_loc]-cint_array[i_loc][0]),(cint_array[i_loc][1] - dataset['alpha'][i_loc])])
                i_loc += 1
            error_arr=np.transpose(np.array(error))

            for j in range(30):
                stderr_alpha.append(dataset['stderr'][j_loc][1]/2)
                j_loc +=1

            estimate = []
            for e in range(30):
                estimate.append(dataset['alpha'][e])
        
        else:
            for i in  dataset['id']:                  #range(dataset['n_local']):
                error.append([(dataset['alpha'][i_loc]-cint_array[i_loc][0]),(cint_array[i_loc][1] - dataset['alpha'][i_loc])])
                i_loc += 1
            error_arr=np.transpose(np.array(error))

            for j in dataset['id']:
                stderr_alpha.append(dataset['stderr'][j_loc][1]/2)
                j_loc +=1
            estimate = dataset['alpha']
        
        x_pos = np.arange(len(estimate))
        x_pos_err = x_pos + 0.15
        x_pos_stderr = x_pos-0.15

        fig, ax = plt.subplots()

        ax.bar(x_pos, estimate, align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.errorbar(x_pos_err, estimate, yerr=error_arr, alpha=0.5, elinewidth=1, linewidth=0, ecolor='black', capsize=1, color='white')
        ax.errorbar(x_pos_stderr, estimate, yerr=stderr_alpha, elinewidth=1, linewidth=0, alpha=0.5, ecolor='blue', capsize=1, color='white')
        ax.set_ylabel('Confidence Interval')
        #ax.set_xticks(x_pos)
        #ax.set_xticklabels(materials)
        #ax.set_title('Bootstrap Confidence intercals')
        ax.yaxis.grid(True)
        
        plt.tight_layout()
        #plt.savefig('bar_plot_with_error_bars.png')

        plt_name_a = param['dataset_dir'] + param['outprefix'] + "_" + str(param['mu']) + "_" + str(param['alpha']) + "_" + str(param['beta'])+"/"
        plt_name_a = plt_name_a + "N_" + str(param['n']) + "_T_" + str(int(param['t'])) + "/" 
        os.makedirs(plt_name_a,  exist_ok=True)
        plt_name_a = plt_name_a + "cint1alpha.png"
        plt.savefig(plt_name_a )
        logger.info("Plot saved in " + str(plt_name_a))
        #plt.show()
        

    
