import numpy as np
import logging
import time
from mpi4py import MPI

from lib import Hawkes as hk
from lib.simulator import mcgen

def bootstrap(param, dataset, comm):

    bt = param['bt']

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
        print("param_bt: ", param_bt)
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
    
