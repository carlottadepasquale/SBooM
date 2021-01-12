# from tick.base import TimeFunction
# from tick.hawkes import SimuHawkesExpKernels
# from tick.plot import plot_point_process
from lib import Hawkes as hk
import numpy as np
import matplotlib.pyplot as plt
import time
import logging
from mpi4py import MPI

def simulator(param, dataset, comm):
    
    logger=logging.getLogger(param["logger"])
    
    n = param["n"]
    s = param["size"]
    r = param["rank"]
    if "seed" in param:
        np.random.seed(param["seed"] + param["rank"])
    else:
        t = int(time.time())
        np.random.seed(param["rank"] + t)
    
    i_local = 0
    t_sim = np.zeros(1)
    t_est = np.zeros(1)
    total_sim = np.zeros(1)
    total_est = np.zeros(1)
    mu_ok = np.zeros(1)
    alpha_ok = np.zeros(1)
    beta_ok = np.zeros(1)
    n_events = np.zeros(1)
    mu_ok_tot = np.zeros(1)
    alpha_ok_tot = np.zeros(1)
    beta_ok_tot = np.zeros(1)
    n_events_tot = np.zeros(1)
    ste_error_count = np.zeros(1)
    ste_error_count_tot = np.zeros(1)
    opt = ['stderr']

    for i in dataset['id']:

        t_sim0 = MPI.Wtime()
        hsim = hawkes(param)
        t_sim[0] += MPI.Wtime() - t_sim0
        n_events[0] += len(hsim)


        t_est0 = MPI.Wtime()
        model = inference(hsim, param, opt) #prima mettevamo anche i, tolto
        t_est[0] += MPI.Wtime() - t_est0

        ste_error_count[0] += model.count_err

        log_output = "iteration " + str(i) + "\n"
        log_output += "parameter: "+ str(model.parameter) + "\n"
        log_output += "branching ratio: " + str(model.br) + "\n" # the branching ratio
        log_output += "log-likelihood:" + str(model.L) + "\n" # the log-likelihood of the estimated parameter values
        logger.debug(log_output)

        dataset['t'].append(hsim)
        dataset['alpha'][i_local] = model.parameter['alpha']
        dataset['beta'][i_local] = model.parameter['beta']
        dataset['mu'][i_local] = model.parameter['mu']
        dataset['stderr'].append(model.stderr)
        
        mu_upper_lim = dataset['mu'][i_local]+1.96*dataset['stderr'][i_local][0]
        mu_lower_lim = dataset['mu'][i_local]-1.96*dataset['stderr'][i_local][0]

        alpha_upper_lim = dataset['alpha'][i_local]+1.96*dataset['stderr'][i_local][1]
        alpha_lower_lim = dataset['alpha'][i_local]-1.96*dataset['stderr'][i_local][1]

        beta_upper_lim = dataset['beta'][i_local]+1.96*dataset['stderr'][i_local][2]
        beta_lower_lim = dataset['beta'][i_local]-1.96*dataset['stderr'][i_local][2]
        
        if (mu_upper_lim >= param['mu'] and mu_lower_lim <= param['mu']):
            mu_ok[0] += 1
        if (alpha_upper_lim >= param['alpha'] and alpha_lower_lim <= param['alpha']):
            alpha_ok[0] += 1
        if (beta_upper_lim >= param['beta'] and beta_lower_lim <= param['beta']):
            beta_ok[0] += 1
        i_local += 1

    comm.Reduce(mu_ok, mu_ok_tot, op=MPI.SUM, root=0)
    comm.Reduce(alpha_ok, alpha_ok_tot, op=MPI.SUM, root=0)
    comm.Reduce(beta_ok, beta_ok_tot, op=MPI.SUM, root=0)
    comm.Reduce(n_events, n_events_tot, op=MPI.SUM, root=0) 
    comm.Reduce(ste_error_count, ste_error_count_tot, op=MPI.SUM, root=0)    
    
    avg_t_sim_loc = t_sim/i_local
    avg_t_est_loc = t_est/i_local

    comm.Reduce(avg_t_sim_loc, total_sim, op=MPI.SUM, root=0)
    comm.Reduce(avg_t_est_loc, total_est, op=MPI.SUM, root=0)

    if param['rank'] == 0:
        avg_n_events_p = n_events_tot / n
        avg_n_events_t = param['mu']*param['t']/(1-param['alpha'])
        
        avg_t_sim = total_sim[0]/param['size']
        avg_t_est = total_est[0]/param['size']

        nevents_format = "{n:.1f}"
        avgt_format = "{n:.5f}"
        log_string = '\n- mu_asymptotic:    ' + str(100*mu_ok_tot[0]/param['n']) + '%\n'
        log_string += '- alpha_asymptotic: ' + str(100*alpha_ok_tot[0]/param['n']) + '%\n'
        log_string += '- beta_asymptotic:  '+ str(100*beta_ok_tot[0]/n) + '%\n'
        log_string += "- Stderr calculation fails: " + str(ste_error_count_tot[0]) + '\n'

        log_string += '- Theoretical avg n of events: ' + nevents_format.format(n=avg_n_events_t) + '\n'
        log_string += '- Avg n of events: ' + nevents_format.format(n=avg_n_events_p[0]) + '\n'

        log_string += ("- Avg time Hawkes: " + avgt_format.format(n=avg_t_sim) +"\n" + "- Avg time inference: "+ avgt_format.format(n=avg_t_est))

        logger.info(log_string)


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
        
    return hsim

def inference(hsim, param, opt):
    model = hk.estimator().set_kernel('exp',num_exp=1).set_baseline('const')
    t = int(param['t'])
    interval = [0,t]
    model.fit(hsim, interval, opt=opt)  #,  opt=["ste", "print"])  #, "check"])
    # stderr = model.stderr
    # print('Std Error: ', stderr)
    return model
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