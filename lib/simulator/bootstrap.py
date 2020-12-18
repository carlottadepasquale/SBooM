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
        for b in range(bt):
            hsim_bt = mcgen.hawkes(param_bt)
            model_bt = mcgen.inference(i, hsim_bt, param)
            dataset['bootstrap'][j_local]['alpha'][b_i] = model_bt.parameter['alpha']
            dataset['bootstrap'][j_local]['beta'][b_i] = model_bt.parameter['beta']
            dataset['bootstrap'][j_local]['mu'][b_i] = model_bt.parameter['mu']
            b_i += 1
        j_local += 1
        #print("dset_bt: ", dataset['bootstrap'])
    
def confidence_int_1(param, dataset, comm):
    
    dataset['cint_alpha_1'] = np.empty(n_local)
    dataset['cint_beta_1'] = np.empty(n_local)
    dataset['cint_mu_1'] = np.empty(n_local)
    
    i_loc = 0

    for i in dataset['id']:
        q1alpha = np.quantile(dataset['bootstrap'][i_loc]['alpha'], 0.05)
        q1beta = np.quantile(dataset['bootstrap'][i_loc]['beta'], 0.05)
        q1mu = np.quantile(dataset['bootstrap'][i_loc]['mu'], 0.05)

        q2alpha = np.quantile(dataset['bootstrap'][i_loc]['alpha'], 0.95)
        q2beta = np.quantile(dataset['bootstrap'][i_loc]['beta'], 0.95)
        q2mu = np.quantile(dataset['bootstrap'][i_loc]['mu'], 0.95)

        i_loc += 1

    print('cint1 a: ', q1alpha, q2alpha)
    print('cint1 b: ', q1beta, q2beta)
    print('cint1 m: ', q1mu, q2mu)



def confidence_int_2(param, dataset, comm):
    
    dataset['cint_alpha_2'] = np.empty(n_local)
    dataset['cint_beta_2'] = np.empty(n_local)
    dataset['cint_mu_2'] = np.empty(n_local)

    for i in dataset['id']:
        q1alpha = np.quantile(dataset['bootstrap'][0]['alpha'], 0.05)
        q1beta = np.quantile(dataset['bootstrap'][0]['beta'], 0.05)
        q1mu = np.quantile(dataset['bootstrap'][0]['mu'], 0.05)

        q2alpha = np.quantile(dataset['bootstrap'][0]['alpha'], 0.95)
        q2beta = np.quantile(dataset['bootstrap'][0]['beta'], 0.95)
        q2mu = np.quantile(dataset['bootstrap'][0]['mu'], 0.95)

        cint2alpha = [2*dataset['alpha'][0]-q2alpha, 2*dataset['alpha'][0]-q1alpha]
        cint2beta = [2*dataset['beta'][0]-q2beta, 2*dataset['beta'][0]-q1beta]
        cint2mu = [2*dataset['mu'][0]-q2mu, 2*dataset['mu'][0]-q1mu]

    print("alpha cint 2: ", cint2alpha)
    print("beta cint 2: ", cint2beta)
    print("mu cint 2: ", cint2mu)

    
def confidence_int_3(param, dataset, comm):
    
    dataset['cint_alpha_3'] = np.empty(n_local)
    dataset['cint_beta_3'] = np.empty(n_local)
    dataset['cint_mu_3'] = np.empty(n_local)
    
    stderr_a_bt, stderr_b_bt, stderr_m_bt = stderr_calc(param, dataset, comm)

    i_loc = 0
    for i in dataset['id']:
        alpha_cint = [dataset['alpha'][i_loc] - 1.96*stderr_a_bt[i_loc], dataset['alpha'][i_loc] + 1.96*stderr_a_bt[i_loc]]
        beta_cint = [dataset['beta'][i_loc] - 1.96*stderr_b_bt[i_loc], dataset['beta'][i_loc] + 1.96*stderr_b_bt[i_loc]]
        mu_cint = [dataset['mu'][i_loc] - 1.96*stderr_m_bt[i_loc], dataset['mu'][i_loc] + 1.96*stderr_m_bt[i_loc]]
        i_loc += 1

    print("alpha cint 3: ", alpha_cint)
    print("beta cint 3: ", beta_cint)
    print("mu cint 3: ", mu_cint)


def confidence_int_4(param, dataset, comm):

    dataset['cint_alpha_4'] = np.empty(n_local)
    dataset['cint_beta_4'] = np.empty(n_local)
    dataset['cint_mu_4'] = np.empty(n_local)

    stderr_a_bt, stderr_b_bt, stderr_m_bt = stderr_calc(param, dataset, comm)

    i_loc = 0

    for i in dataset['id']:
        standardized_alpha_bt = (dataset['bootstrap'][i_loc]['alpha']-dataset['alpha'][i_loc])/stderr_a_bt[i_loc]
        standardized_beta_bt = (dataset['bootstrap'][0]['beta']-dataset['beta'][0])/stderr_b_bt[i_loc]
        standardized_mu_bt = (dataset['bootstrap'][0]['mu']-dataset['mu'][0])/stderr_m_bt[i_loc]

        q1alpha = np.quantile(standardized_alpha_bt[i_loc], 0.05)
        q1beta = np.quantile(standardized_beta_bt[i_loc], 0.05)
        q1mu = np.quantile(standardized_mu_bt[i_loc], 0.05)

        q2alpha = np.quantile(standardized_alpha_bt[i_loc], 0.95)
        q2beta = np.quantile(standardized_beta_bt[i_loc], 0.95)
        q2mu = np.quantile(standardized_mu_bt[i_loc], 0.95)

        cint4alpha = [dataset['alpha'][0]-q2alpha*stderr_a_bt, dataset['alpha'][0]-q1alpha*stderr_a_bt]
        cint4beta = [dataset['beta'][0]-q2beta*stderr_b_bt, dataset['beta'][0]-q1beta*stderr_b_bt]
        cint4mu = [dataset['mu'][0]-q2mu*stderr_m_bt, dataset['mu'][0]-q1mu*stderr_m_bt]

        i_loc += 1

    print("alpha cint 4: ", cint4alpha)
    print("beta cint 4: ", cint4beta)
    print("mu cint 4: ", cint4mu)

    
def stderr_calc(param, dataset, comm):
    i_loc = 0
    for i in dataset['id']:
        stderr_a_bt = np.empty(dataset['n_local'])
        stderr_b_bt = np.empty(dataset['n_local'])
        stderr_m_bt = np.empty(dataset['n_local'])
        stderr_a_bt[i_loc] = np.std(dataset['bootstrap'][i_loc]['alpha'], ddof=1) / np.sqrt(param['bt'])
        stderr_b_bt[i_loc] = np.std(dataset['bootstrap'][i_loc]['beta'], ddof=1) / np.sqrt(param['bt'])
        stderr_m_bt[i_loc] = np.std(dataset['bootstrap'][i_loc]['mu'], ddof=1) / np.sqrt(param['bt'])
        i_loc += 1

    return [stderr_a_bt, stderr_b_bt, stderr_m_bt]
