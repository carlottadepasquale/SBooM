import numpy as np

def init_dataset(param):
    mc_dataset = {}
    mc_dataset['rank'] = param["rank"]
    
    rank_map,n_local,id_array = id_map(param)

    mc_dataset['rank_map'] = rank_map 
    mc_dataset['id'] = id_array 
    
    mc_dataset['alpha'] = np.empty(n_local)
    mc_dataset['beta'] = np.empty(n_local)
    mc_dataset['mu'] = np.empty(n_local)
    mc_dataset['stderr'] =[] 
    mc_dataset['t'] = []
    mc_dataset['n_it'] = param['n']
    mc_dataset['n_local'] = n_local
    mc_dataset['bootstrap'] = []
    return mc_dataset

def id_map(param):

    rank_map =  get_rank_map(param)  
    i_start = rank_map[param['rank']]['i_start'] 
    i_end = rank_map[param['rank']]['i_end']
    n_local = i_end - i_start
    id_array = np.arange(i_start, i_end )
    return [rank_map,n_local,id_array]

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
