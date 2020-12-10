import numpy as np

def init_dataset(param):
    mc_dataset = {}
    mc_dataset['rank'] = param["rank"]
    mc_dataset['rank_map'] =  get_rank_map(param)  #i_end e i_start di ogni processore
    i_start = mc_dataset['rank_map'][param['rank']]['i_start'] 
    i_end = mc_dataset['rank_map'][param['rank']]['i_end']
    n_local = i_end - i_start
    mc_dataset['alpha'] = np.empty(n_local)
    mc_dataset['beta'] = np.empty(n_local)
    mc_dataset['mu'] = np.empty(n_local)
    mc_dataset['stderr'] =[] 
    mc_dataset['t'] = []
    mc_dataset['n_it'] = param['n']
    mc_dataset['n_local'] = n_local
    mc_dataset['id'] = np.arange(i_start, i_end )
    mc_dataset['bootstrap'] = []
    return mc_dataset

def get_rank_map(param):
    n = param["n"]
    size = param["size"]
    rank_map = []
    for r in range(size):
        iterations = param["n"]//param["size"]
        i_start = r * iterations
        i_end = i_start + iterations
        if param["rank"] == param["size"]-1:
            i_end +=  param["n"]%param["size"]
        rank_map.append({'i_start': i_start, 'i_end': i_end})
    return rank_map
