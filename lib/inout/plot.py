#
# BootMCHawkes
#
# @authors : Carlotta De Pasquale

from lib import Hawkes as hk
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
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
    
    if param['rank']==0: 
        count = np.zeros(param['size'])
        displ = np.zeros(param['size'])
        c = 0
        for i in range(param['size']):
            count[c] = dataset['n_local']
            displ[c] = dataset['id'][0]
            c += 1
        #print(param['rank'], ' count: ', count, ' displ: ', displ)


    comm.Gather(dataset['alpha'], alpha_all, root = 0)
    comm.Gather(dataset['beta'], beta_all, root = 0)
    comm.Gather(dataset['mu'], mu_all, root = 0)
    
    if param['rank'] == 0:
        #print("alpha_all: ", alpha_all)
        sns.distplot(alpha_all, hist=False, kde=True, color = 'darkblue', kde_kws={'linewidth': 4})
        plt.show()

    
