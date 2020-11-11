from tick.base import TimeFunction
from tick.hawkes import SimuHawkesExpKernels
from tick.plot import plot_point_process
import numpy as np
import matplotlib.pyplot as plt

def simulator(param):
    #iterations = n//size
    #if rank == size-1:
    #    iterations = n//size + n%size
    hawkes(param)
    
def hawkes(param):
    
    adjacency = np.array([[param['a']]])
    decay = np.array([[param['d']]])
    baseline = np.array([param['m']])
    t = int(param['t'])
    hawkes = SimuHawkesExpKernels(adjacency, decay, baseline)
    hawkes.track_intensity(0.1)
    hawkes.end_time = t
    hawkes.simulate()
    plot(hawkes)
    
def plot(hawkes_obj):
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    plot_point_process(hawkes_obj, ax=ax)

    t_values = np.linspace(0, hawkes_obj.end_time, 100)
    ax.plot(t_values, hawkes_obj.get_baseline_values(0, t_values), label='baseline',
            ls='--', lw=1)
    ax.set_ylabel("$\lambda(t)$", fontsize=18)
    ax.legend()

    plt.title("Intensity Hawkes process with exponential kernel")
    fig.tight_layout()
    plt.show()