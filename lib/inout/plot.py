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

def plot_process(dataset, model):
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
        beta_sort = np.sort(beta_all)
        logger.debug("beta_max" + str(beta_sort[-10:]))
        max_alpha = np.max(alpha_all)
        max_beta = np.max(beta_all)
        max_mu = np.max(mu_all)
        min_alpha = np.min(alpha_all)
        min_beta = np.min(beta_all)
        min_mu = np.min(mu_all)
        logger.debug("Alpha, beta, mu max: " + str(max_alpha) + "   " +str(max_beta)+  "   " +str(max_mu))
        logger.debug("Alpha, beta, mu min: " + str(min_alpha) + str(min_beta) + str(min_mu))
        
        fig, ax =plt.subplots(1,3)
        plt.subplots_adjust(wspace = 0.35)
        a = sns.histplot(alpha_all, ax=ax[0])
        b = sns.histplot(beta_all, ax=ax[1], color='lightcoral')
        m = sns.histplot(mu_all, ax=ax[2], color='mediumpurple')
        vlines = [param['alpha'], param['beta'], param['mu']]
        v=0
        for ax in fig.axes:
            ax.axvline(vlines[v], color='red')
            v += 1
        a.set_title('alpha')
        b.set_title('beta')
        m.set_title('mu')
        plt_name = param['dataset_dir'] + param['outprefix'] + "_" + str(param['mu']) + "_" + str(param['alpha']) + "_" + str(param['beta'])+"/"
        plt_name = plt_name + "N_" + str(param['n']) + "_T_" + str(int(param['t'])) + "/" 
        os.makedirs(plt_name,  exist_ok=True)
        plt_name = plt_name + "allplts.png"
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

    if param['rank'] == 2:
        ######################### ALPHA ###########################
        error_alpha=[]
        stderr_alpha = []
        cint_array = np.array(dataset['cint_alpha_1'])
        i_loc = 0
        j_loc = 0

        if dataset['n_local'] >80:
            for i in range(80):                  #range(dataset['n_local']):
                error_alpha.append([(dataset['alpha'][i_loc]-cint_array[i_loc][0]),(cint_array[i_loc][1] - dataset['alpha'][i_loc])])
                i_loc += 1
            error_a_arr=np.transpose(np.array(error_alpha))

            for j in range(80):
                stderr_alpha.append(1.96*dataset['stderr'][j_loc][1])
                j_loc +=1

            estimate_a = []
            for e in range(80):
                estimate_a.append(dataset['alpha'][e])
        
        else:
            for i in  dataset['id']:                  #range(dataset['n_local']):
                error_alpha.append([(dataset['alpha'][i_loc]-cint_array[i_loc][0]),(cint_array[i_loc][1] - dataset['alpha'][i_loc])])
                i_loc += 1
            error_a_arr=np.transpose(np.array(error))

            for j in dataset['id']:
                stderr_alpha.append(1.96*dataset['stderr'][j_loc][1])
                j_loc +=1
            estimate_a = dataset['alpha']
        
        x_pos = np.arange(len(estimate_a))
        x_pos_err = x_pos + 0.15
        x_pos_stderr = x_pos-0.15

        #print('stderr', stderr_alpha)
        # print('cint', error_arr)

        fig, ax = plt.subplots(figsize=(20,10))

        ax.bar(x_pos, estimate_a, align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.errorbar(x_pos_err, estimate_a, yerr=error_a_arr, alpha=0.5, elinewidth=1.5, linewidth=0, ecolor='black', capsize=3, color='white')
        ax.errorbar(x_pos_stderr, estimate_a, yerr=stderr_alpha, elinewidth=1.5, linewidth=0, alpha=0.5, ecolor='blue', capsize=3, color='white')
        ax.hlines(param['alpha'], -0.5, 80, color='crimson')
        #ax.set_ylabel('Estimates and Confidence Intervals', fontsize=35)
        ax.set_xlabel('Monte Carlo Iteration', fontsize=25)
        ax.tick_params(axis="y", labelsize=20)
        #ax.set_yticklables(ylables, fontsize=8)
        #ax.set_xticks(x_pos)
        #ax.set_xticklabels(materials)
        ax.set_title('Bootstrap vs Asymptotic Confidence intervals, α', fontsize=30)
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

        ############################## BETA ##############################

        error_beta=[]
        stderr_beta = []
        cint_array = np.array(dataset['cint_beta_1'])
        i_loc = 0
        j_loc = 0

        if dataset['n_local'] >80:
            for i in range(80):                  #range(dataset['n_local']):
                error_beta.append([(dataset['beta'][i_loc]-cint_array[i_loc][0]),(cint_array[i_loc][1] - dataset['beta'][i_loc])])
                i_loc += 1
            error_b_arr=np.transpose(np.array(error_beta))

            for j in range(80):
                stderr_beta.append(1.96*dataset['stderr'][j_loc][2])
                j_loc +=1

            estimate_b = []
            for e in range(80):
                estimate_b.append(dataset['beta'][e])
        
        else:
            for i in  dataset['id']:                
                error_beta.append([(dataset['beta'][i_loc]-cint_array[i_loc][0]),(cint_array[i_loc][1] - dataset['beta'][i_loc])])
                i_loc += 1
            error_b_arr=np.transpose(np.array(error_b))

            for j in dataset['id']:
                stderr_beta.append(1.96*dataset['stderr'][j_loc][2])
                j_loc +=1
            estimate_b = dataset['beta']
        
        x_pos = np.arange(len(estimate_b))
        x_pos_err = x_pos + 0.15
        x_pos_stderr = x_pos-0.15

        #print('stderr', stderr_alpha)
        # print('cint', error_arr)

        fig, bx = plt.subplots(figsize=(20,10))

        bx.bar(x_pos, estimate_b, align='center', alpha=0.5, ecolor='black', capsize=10, color='lightcoral')
        bx.errorbar(x_pos_err, estimate_b, yerr=error_b_arr, alpha=0.5, elinewidth=1.5, linewidth=0, ecolor='black', capsize=3, color='white')
        bx.errorbar(x_pos_stderr, estimate_b, yerr=stderr_beta, elinewidth=1.5, linewidth=0, alpha=0.5, ecolor='blue', capsize=3, color='white')
        bx.hlines(param['beta'], -0.5, 80, color='crimson')
        #ax.set_ylabel('Estimates and Confidence Intervals', fontsize=35)
        bx.set_xlabel('Monte Carlo Iteration', fontsize=25)
        bx.tick_params(axis="y", labelsize=20)
        #ax.set_yticklables(ylables, fontsize=8)
        #ax.set_xticks(x_pos)
        #ax.set_xticklabels(materials)
        bx.set_title('Bootstrap vs Asymptotic Confidence intervals, β', fontsize=30)
        bx.yaxis.grid(True)
        
        plt.tight_layout()
        #plt.savefig('bar_plot_with_error_bars.png')

        plt_name_b = param['dataset_dir'] + param['outprefix'] + "_" + str(param['mu']) + "_" + str(param['alpha']) + "_" + str(param['beta'])+"/"
        plt_name_b = plt_name_b + "N_" + str(param['n']) + "_T_" + str(int(param['t'])) + "/" 
        os.makedirs(plt_name_b,  exist_ok=True)
        plt_name_b = plt_name_b + "cint1beta.png"
        plt.savefig(plt_name_b)
        logger.info("Plot saved in " + str(plt_name_b))
        #plt.show()

        ######################## MU ############################

        error_mu=[]
        stderr_mu = []
        cint_array = np.array(dataset['cint_mu_1'])
        i_loc = 0
        j_loc = 0

        if dataset['n_local'] >80:
            for i in range(80):                  #range(dataset['n_local']):
                error_mu.append([(dataset['mu'][i_loc]-cint_array[i_loc][0]),(cint_array[i_loc][1] - dataset['mu'][i_loc])])
                i_loc += 1
            error_m_arr=np.transpose(np.array(error_mu))

            for j in range(80):
                stderr_mu.append(1.96*dataset['stderr'][j_loc][0])
                j_loc +=1

            estimate_m = []
            for e in range(80):
                estimate_m.append(dataset['mu'][e])
        
        else:
            for i in  dataset['id']:                  #range(dataset['n_local']):
                error_mu.append([(dataset['mu'][i_loc]-cint_array[i_loc][0]),(cint_array[i_loc][1] - dataset['mu'][i_loc])])
                i_loc += 1
            error_m_arr=np.transpose(np.array(error_mu))

            for j in dataset['id']:
                stderr_mu.append(1.96*dataset['stderr'][j_loc][0])
                j_loc +=1
            estimate_a = dataset['mu']
        
        x_pos = np.arange(len(estimate_m))
        x_pos_err = x_pos + 0.15
        x_pos_stderr = x_pos-0.15

        #print('stderr', stderr_alpha)
        # print('cint', error_arr)

        fig, mx = plt.subplots(figsize=(20,10))

        mx.bar(x_pos, estimate_m, align='center', alpha=0.5, ecolor='black', capsize=10, color='mediumpurple')
        mx.errorbar(x_pos_err, estimate_m, yerr=error_m_arr, alpha=0.5, elinewidth=1.5, linewidth=0, ecolor='black', capsize=3, color='white')
        mx.errorbar(x_pos_stderr, estimate_m, yerr=stderr_mu, elinewidth=1.5, linewidth=0, alpha=0.5, ecolor='blue', capsize=3, color='white')
        mx.hlines(param['mu'], -0.5, 80, color='crimson')
        #ax.set_ylabel('Estimates and Confidence Intervals', fontsize=35)
        mx.set_xlabel('Monte Carlo Iteration', fontsize=25)
        mx.tick_params(axis="y", labelsize=20)
        #ax.set_yticklables(ylables, fontsize=8)
        #ax.set_xticks(x_pos)
        #ax.set_xticklabels(materials)
        mx.set_title('Bootstrap vs Asymptotic Confidence intervals, μ', fontsize=30)
        mx.yaxis.grid(True)
        
        plt.tight_layout()
        #plt.savefig('bar_plot_with_error_bars.png')

        plt_name_m = param['dataset_dir'] + param['outprefix'] + "_" + str(param['mu']) + "_" + str(param['alpha']) + "_" + str(param['beta'])+"/"
        plt_name_m = plt_name_m + "N_" + str(param['n']) + "_T_" + str(int(param['t'])) + "/" 
        os.makedirs(plt_name_m,  exist_ok=True)
        plt_name_m = plt_name_m + "cint1mu.png"
        plt.savefig(plt_name_m)
        logger.info("Plot saved in " + str(plt_name_m))
        #plt.show()

        
        

    
