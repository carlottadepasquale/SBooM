# from tick.base import TimeFunction
# from tick.hawkes import SimuHawkesExpKernels
# from tick.plot import plot_point_process
from lib import Hawkes as hk
import numpy as np
import matplotlib.pyplot as plt
import time
import logging

def simulator(param, dataset):
    n = param["n"]
    s = param["size"]
    r = param["rank"]
    if "seed" in param:
        np.random.seed(param["seed"] + param["rank"])
    else:
        t = int(time.time())
        np.random.seed(param["rank"] + t)
    i_local = 0
    for i in dataset['id']:
        hsim = hawkes(param)
        estimate = inference(i, hsim, param)
        dataset['t'].append(hsim)
        dataset['a'][i_local] = estimate['alpha']
        dataset['b'][i_local] = estimate['beta']
        dataset['m'][i_local] = estimate['mu']
        i_local += 1
    print(str(dataset))
    
def hawkes(param):
    
    adjacency = np.array([[param['alpha']]])
    decay = np.array([[param['beta']]])
    baseline = np.array([param['mu']])
    t = int(param['t'])
    model = hk.simulator()
    model.set_kernel('exp')
    model.set_baseline('const')
    par = {'mu': baseline, 'alpha': adjacency, 'beta': decay}
    model.set_parameter(par)
    interval = [0,t]
    hsim = model.simulate(interval)
    if "plt" in param["execution"]:
        model.plot_l()
        model.plot_N()
        plt.show()
    return hsim

def inference(i, hsim, param):
    model = hk.estimator().set_kernel('exp',num_exp=1).set_baseline('const')
    t = int(param['t'])
    interval = [0,t]
    model.fit(hsim, interval)
    log_output = "iteration " + str(i) + "\n"
    log_output += "parameter: "+ str(model.parameter) + "\n"
    log_output += "branching ratio: " + str(model.br) + "\n" # the branching ratio
    log_output += "log-likelihood:" + str(model.L) + "\n" # the log-likelihood of the estimated parameter values
    logger=logging.getLogger(param["logger"])
    logger.info(log_output)
    return model.parameter
    # # T_trans: a list of transformed event occurrence times, itv_trans: the transformed observation interval
    # [T_trans, itv_trans] = model.t_trans() 
    # # Kormogorov-Smirnov test under the null hypothesis that the transformed event occurrence times are uniformly distributed
    # model.plot_KS()
    # # hawkes = SimuHawkesExpKernels(adjacency, decay, baseline)
    # # hawkes.track_intensity(0.1)
    # # hawkes.end_time = t
    # # hawkes.simulate()
    # # plot(hawkes)
    
# def plot(hawkes_obj):
#     fig, ax = plt.subplots(1, 1, figsize=(10, 4))

#     plot_point_process(hawkes_obj, ax=ax)

#     t_values = np.linspace(0, hawkes_obj.end_time, 100)
#     ax.plot(t_values, hawkes_obj.get_baseline_values(0, t_values), label='baseline',
#             ls='--', lw=1)
#     ax.set_ylabel("$\lambda(t)$", fontsize=18)
#     ax.legend()

#     plt.title("Intensity Hawkes process with exponential kernel")
#     fig.tight_layout()
#     plt.show()