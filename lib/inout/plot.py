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
        print(param['rank'], ' count: ', count, ' displ: ', displ)
    
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
        print("Alpha max: ", max_alpha)
        print("Alpha, beta, mu min: ", min_alpha, min_beta, min_mu)
        
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
        print("Plot saved in ", plt_name)
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
        

    
