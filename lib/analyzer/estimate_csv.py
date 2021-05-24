from lib import Hawkes as hk
from lib.simulator import mcgen
from lib.simulator import bootstrap

import numpy as np
import matplotlib.pyplot as plt
import time
import logging
from mpi4py import MPI

def estimate_exp(param, dataset):
    hsim = dataset['t'][0]
    opt = ['stderr'] #'check'] #, 'print', 'check']
    model = mcgen.inference(hsim, param, opt)
    dataset['alpha'][0] = model.parameter['alpha']
    dataset['beta'][0] = model.parameter['beta']
    dataset['mu'][0] = model.parameter['mu']
    dataset['stderr'].append(model.stderr)
    print('alpha est: ', dataset['alpha'][0])
    print('beta est: ', dataset['beta'][0])
    print('mu est: ', dataset['mu'][0])

def bootstrap_cint(param, dataset, comm):
    #logger=logging.getLogger(param["logger"]) 
    bootstrap.bootstrap(param, dataset, comm)
    #print('bt: ', dataset['bootstrap'])
    bootstrap.confidence_int_1(param, dataset, comm)
    bootstrap.confidence_int_2(param, dataset, comm)
    bootstrap.confidence_int_3(param, dataset, comm)
    bootstrap.confidence_int_4(param, dataset, comm)
    bootstrap.confidence_int_5(param, dataset, comm)
