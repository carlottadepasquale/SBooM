import numpy as np
import logging
import time
from mpi4py import MPI

from lib import Hawkes as hk
from lib.simulator import mcgen

def bootstrap(param, dataset, comm):

    bt = param['bt']
    logger=logging.getLogger(param["logger"])

    if "seed" in param:
        np.random.seed(param["seed"] + param["rank"])
    else:
        t = int(time.time())
        np.random.seed(param["rank"] + t)

    j_local = 0
    for i in dataset['id']:
        param_bt = {}
        param_bt['mu'] = dataset['mu'][j_local]
        param_bt['alpha'] = dataset['alpha'][j_local]
        param_bt['beta'] = dataset['beta'][j_local]
        param_bt['t'] = param['t']
        logger.debug("param_bt: " + str(param_bt))
        b_i = 0
        dataset['bootstrap'].append({'alpha': np.zeros(bt), 'beta': np.zeros(bt) , 'mu': np.zeros(bt)}) 
        opt = ['nostderr']
        for b in range(bt):
            hsim_bt = mcgen.hawkes(param_bt)
            model_bt = mcgen.inference(hsim_bt, param, opt)
            dataset['bootstrap'][j_local]['alpha'][b_i] = model_bt.parameter['alpha']
            dataset['bootstrap'][j_local]['beta'][b_i] = model_bt.parameter['beta']
            dataset['bootstrap'][j_local]['mu'][b_i] = model_bt.parameter['mu']
            b_i += 1
        j_local += 1
        #print("dset_bt: ", dataset['bootstrap'])
    
def confidence_int_1(param, dataset, comm):

    dataset['cint_alpha_1'] = []
    dataset['cint_beta_1'] = []
    dataset['cint_mu_1'] = []
    
    i_loc = 0
    alpha_1_ok = np.zeros(1)
    beta_1_ok = np.zeros(1)
    mu_1_ok = np.zeros(1)
    mu_1_ok_tot = np.zeros(1)
    alpha_1_ok_tot = np.zeros(1)
    beta_1_ok_tot = np.zeros(1)

    for i in dataset['id']:
        q1alpha = np.quantile(dataset['bootstrap'][i_loc]['alpha'], 0.05)
        q1beta = np.quantile(dataset['bootstrap'][i_loc]['beta'], 0.05)
        q1mu = np.quantile(dataset['bootstrap'][i_loc]['mu'], 0.05)

        q2alpha = np.quantile(dataset['bootstrap'][i_loc]['alpha'], 0.95)
        q2beta = np.quantile(dataset['bootstrap'][i_loc]['beta'], 0.95)
        q2mu = np.quantile(dataset['bootstrap'][i_loc]['mu'], 0.95)

        dataset['cint_alpha_1'].append([q1alpha, q2alpha])
        dataset['cint_beta_1'].append([q1beta, q2beta])
        dataset['cint_mu_1'].append([q1mu, q2mu])

        if (q2mu >= param['mu'] and q1mu <= param['mu']):
            mu_1_ok[0] += 1
        if (q2alpha >= param['alpha'] and q1alpha <= param['alpha']):
            alpha_1_ok[0] += 1
        if (q2beta >= param['beta'] and q1beta <= param['beta']):
            beta_1_ok[0] += 1

        i_loc += 1
    
    comm.Reduce(mu_1_ok, mu_1_ok_tot, op=MPI.SUM, root=0)
    comm.Reduce(alpha_1_ok, alpha_1_ok_tot, op=MPI.SUM, root=0)
    comm.Reduce(beta_1_ok, beta_1_ok_tot, op=MPI.SUM, root=0) 

    # print("alpha cint 1: ", dataset['cint_alpha_1'])
    # print("beta cint 1: ", dataset['cint_beta_1'])
    # print("mu cint 1: ", dataset['cint_mu_1'])

    if param['rank'] ==0:
        print('alpha_1_ok: ', alpha_1_ok_tot/dataset['n_it']*100, '%')
        print('beta_1_ok: ', beta_1_ok_tot/dataset['n_it']*100, '%')
        print('mu_1_ok: ', mu_1_ok_tot/dataset['n_it']*100, '%')

def confidence_int_2(param, dataset, comm):
    
    dataset['cint_alpha_2'] = []
    dataset['cint_beta_2'] = []
    dataset['cint_mu_2'] = []

    alpha_2_ok = np.zeros(1)
    beta_2_ok = np.zeros(1)
    mu_2_ok = np.zeros(1)
    mu_2_ok_tot = np.zeros(1)
    alpha_2_ok_tot = np.zeros(1)
    beta_2_ok_tot = np.zeros(1)

    i_loc = 0
    for i in dataset['id']:
        
        q1alpha = np.quantile(dataset['bootstrap'][i_loc]['alpha'], 0.05)
        q1beta = np.quantile(dataset['bootstrap'][i_loc]['beta'], 0.05)
        q1mu = np.quantile(dataset['bootstrap'][i_loc]['mu'], 0.05)

        q2alpha = np.quantile(dataset['bootstrap'][i_loc]['alpha'], 0.95)
        q2beta = np.quantile(dataset['bootstrap'][i_loc]['beta'], 0.95)
        q2mu = np.quantile(dataset['bootstrap'][i_loc]['mu'], 0.95)

        dataset['cint_alpha_2'].append([2*dataset['alpha'][i_loc]-q2alpha, 2*dataset['alpha'][i_loc]-q1alpha])
        dataset['cint_beta_2'].append([2*dataset['beta'][i_loc]-q2beta, 2*dataset['beta'][i_loc]-q1beta])
        dataset['cint_mu_2'].append([2*dataset['mu'][i_loc]-q2mu, 2*dataset['mu'][i_loc]-q1mu])

        if ( dataset['cint_mu_2'][-1][1]>= param['mu'] and dataset['cint_mu_2'][-1][0] <= param['mu']):
            mu_2_ok[0] += 1
        if (dataset['cint_mu_2'][-1][1] >= param['alpha'] and dataset['cint_alpha_2'][-1][0] <= param['alpha']):
            alpha_2_ok[0] += 1
            #print('alpha: ', dataset['bootstrap'][i_loc]['alpha'])
        if (dataset['cint_beta_2'][-1][1] >= param['beta'] and dataset['cint_beta_2'][-1][0] <= param['beta']):
            beta_2_ok[0] += 1

        i_loc += 1

    
    comm.Reduce(mu_2_ok, mu_2_ok_tot, op=MPI.SUM, root=0)
    comm.Reduce(alpha_2_ok, alpha_2_ok_tot, op=MPI.SUM, root=0)
    comm.Reduce(beta_2_ok, beta_2_ok_tot, op=MPI.SUM, root=0) 

    # print("alpha cint 2: ", dataset['cint_alpha_2'])
    # print("beta cint 2: ", dataset['cint_beta_2'])
    # print("mu cint 2: ", dataset['cint_mu_2'])

    if param['rank'] ==0:
        print('alpha_2_ok: ', alpha_2_ok_tot/dataset['n_it']*100, '%')
        print('beta_2_ok: ', beta_2_ok_tot/dataset['n_it']*100, '%')
        print('mu_2_ok: ', mu_2_ok_tot/dataset['n_it']*100, '%')

    
def confidence_int_3(param, dataset, comm):
    
    dataset['cint_alpha_3'] = []
    dataset['cint_beta_3'] = []
    dataset['cint_mu_3'] = []
    
    stderr_a_bt, stderr_b_bt, stderr_m_bt = stderr_calc(param, dataset, comm)
    
    alpha_3_ok = np.zeros(1)
    beta_3_ok = np.zeros(1)
    mu_3_ok = np.zeros(1)
    mu_3_ok_tot = np.zeros(1)
    alpha_3_ok_tot = np.zeros(1)
    beta_3_ok_tot = np.zeros(1)
    i_loc = 0
    for i in dataset['id']:
        
        q1alpha = dataset['alpha'][i_loc] - 1.96*stderr_a_bt[i_loc]
        q2alpha = dataset['alpha'][i_loc] + 1.96*stderr_a_bt[i_loc]
        q1beta = dataset['beta'][i_loc] - 1.96*stderr_b_bt[i_loc]
        q2beta = dataset['beta'][i_loc] + 1.96*stderr_b_bt[i_loc]
        q1mu = dataset['mu'][i_loc] - 1.96*stderr_m_bt[i_loc]
        q2mu = dataset['mu'][i_loc] + 1.96*stderr_m_bt[i_loc]

        dataset['cint_alpha_3'].append([q1alpha, q2alpha])
        dataset['cint_beta_3'].append([q1beta, q2beta])
        dataset['cint_mu_3'].append([q1mu, q2mu ])
        
        if (q2mu >= param['mu'] and q1mu <= param['mu']):
            mu_3_ok[0] += 1
        if (q2alpha >= param['alpha'] and q1alpha <= param['alpha']):
            #print(i, 'alpha: ', dataset['bootstrap'][i_loc]['alpha'], "confidence int: ", dataset['cint_alpha_3'][-1])
            alpha_3_ok[0] += 1
        if (q2beta >= param['beta'] and q1beta <= param['beta']):
            beta_3_ok[0] += 1
        
        i_loc += 1
    
    comm.Reduce(mu_3_ok, mu_3_ok_tot, op=MPI.SUM, root=0)
    comm.Reduce(alpha_3_ok, alpha_3_ok_tot, op=MPI.SUM, root=0)
    comm.Reduce(beta_3_ok, beta_3_ok_tot, op=MPI.SUM, root=0) 

    # print("alpha cint 3: ", dataset['cint_alpha_3'])
    # print("beta cint 3: ", dataset['cint_beta_3'])
    # print("mu cint 3: ", dataset['cint_mu_3'])

    if param['rank'] ==0:
        print('alpha_3_ok: ', alpha_3_ok_tot/dataset['n_it']*100, '%')
        print('beta_3_ok: ', beta_3_ok_tot/dataset['n_it']*100, '%')
        print('mu_3_ok: ', mu_3_ok_tot/dataset['n_it']*100, '%')


def confidence_int_4(param, dataset, comm):
    
    dataset['cint_alpha_4'] = []
    dataset['cint_beta_4'] = []
    dataset['cint_mu_4'] = []

    stderr_a_bt, stderr_b_bt, stderr_m_bt = stderr_calc(param, dataset, comm)

    alpha_4_ok = np.zeros(1)
    beta_4_ok = np.zeros(1)
    mu_4_ok = np.zeros(1)
    mu_4_ok_tot = np.zeros(1)
    alpha_4_ok_tot = np.zeros(1)
    beta_4_ok_tot = np.zeros(1)
    i_loc = 0
    #print(dataset['id'][0], 'alpha: ', dataset['bootstrap'][i_loc]['alpha'], 'beta: ', dataset['bootstrap'][i_loc]['beta'], 'mu: ', dataset['bootstrap'][i_loc]['mu'])
    for i in dataset['id']:
        standardized_alpha_bt = (dataset['bootstrap'][i_loc]['alpha'][:]-dataset['alpha'][i_loc])/stderr_a_bt[i_loc]
        standardized_beta_bt = (dataset['bootstrap'][i_loc]['beta'][:]-dataset['beta'][i_loc])/stderr_b_bt[i_loc]
        standardized_mu_bt = (dataset['bootstrap'][i_loc]['mu'][:]-dataset['mu'][i_loc])/stderr_m_bt[i_loc]
        
        #print('std_alpha:', standardized_alpha_bt)

        q1alpha = np.quantile(standardized_alpha_bt, 0.05)
        q1beta = np.quantile(standardized_beta_bt, 0.05)
        q1mu = np.quantile(standardized_mu_bt, 0.05)

        q2alpha = np.quantile(standardized_alpha_bt, 0.95)
        q2beta = np.quantile(standardized_beta_bt, 0.95)
        q2mu = np.quantile(standardized_mu_bt, 0.95)
        # print('q1a:', q2alpha)
        # print('q1b:', q2beta)
        # print('q1m:', q2mu)

        dataset['cint_alpha_4'].append([dataset['alpha'][i_loc]-q2alpha*stderr_a_bt[i_loc], dataset['alpha'][i_loc]-q1alpha*stderr_a_bt[i_loc]])
        dataset['cint_beta_4'].append([dataset['beta'][i_loc]-q2beta*stderr_b_bt[i_loc], dataset['beta'][i_loc]-q1beta*stderr_b_bt[i_loc]])
        dataset['cint_mu_4'].append([dataset['mu'][i_loc]-q2mu*stderr_m_bt[i_loc], dataset['mu'][i_loc]-q1mu*stderr_m_bt[i_loc]])

        if (dataset['cint_mu_4'][-1][1] >= param['mu'] and dataset['cint_mu_4'][-1][0] <= param['mu']):
            mu_4_ok[0] += 1
        if (dataset['cint_alpha_4'][-1][1] >= param['alpha'] and dataset['cint_alpha_4'][-1][0] <= param['alpha']):
            alpha_4_ok[0] += 1
        if (dataset['cint_beta_4'][-1][1] >= param['beta'] and dataset['cint_beta_4'][-1][0] <= param['beta']):
            beta_4_ok[0] += 1

        i_loc += 1

    comm.Reduce(mu_4_ok, mu_4_ok_tot, op=MPI.SUM, root=0)
    comm.Reduce(alpha_4_ok, alpha_4_ok_tot, op=MPI.SUM, root=0)
    comm.Reduce(beta_4_ok, beta_4_ok_tot, op=MPI.SUM, root=0)

    # print("alpha cint 4: ", dataset['cint_alpha_4'])
    # print("beta cint 4: ", dataset['cint_beta_4'])
    # print("mu cint 4: ", dataset['cint_mu_4'])

    if param['rank'] ==0:
        print('alpha_4_ok: ', alpha_4_ok_tot/dataset['n_it']*100, '%')
        print('beta_4_ok: ', beta_4_ok_tot/dataset['n_it']*100, '%')
        print('mu_4_ok: ', mu_4_ok_tot/dataset['n_it']*100, '%')

    
def stderr_calc(param, dataset, comm):
    i_loc = 0
    stderr_a_bt = np.empty(dataset['n_local'])
    stderr_b_bt = np.empty(dataset['n_local'])
    stderr_m_bt = np.empty(dataset['n_local'])
    for i in dataset['id']:
        stderr_a_bt[i_loc] = np.std(dataset['bootstrap'][i_loc]['alpha'], ddof=1) / np.sqrt(param['bt'])
        stderr_b_bt[i_loc] = np.std(dataset['bootstrap'][i_loc]['beta'], ddof=1) / np.sqrt(param['bt'])
        stderr_m_bt[i_loc] = np.std(dataset['bootstrap'][i_loc]['mu'], ddof=1) / np.sqrt(param['bt'])
        i_loc += 1

    return [stderr_a_bt, stderr_b_bt, stderr_m_bt]
