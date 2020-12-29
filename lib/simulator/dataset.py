import numpy as np

def init_dataset(param):
    mc_dataset = {}
    mc_dataset['rank'] = param["rank"]
    
    rank_map,n_local,id_array = id_map(param["rank"],param["size"],param["n"])

    mc_dataset['rank_map'] = rank_map 
    mc_dataset['id'] = id_array 
    
    mc_dataset['alpha'] = np.zeros(n_local)
    mc_dataset['beta'] = np.zeros(n_local)
    mc_dataset['mu'] = np.zeros(n_local)
    mc_dataset['stderr'] =[] 
    mc_dataset['t'] = []
    mc_dataset['n_it'] = param['n']
    mc_dataset['n_local'] = n_local
    mc_dataset['bootstrap'] = []
    
    return mc_dataset

def id_map(rank,size,n):

    rank_map =  get_rank_map(size,n)  
    i_start = rank_map[rank]['i_start'] 
    i_end = rank_map[rank]['i_end']
    n_local = i_end - i_start
    id_array = np.arange(i_start, i_end )
    return [rank_map,n_local,id_array]

def get_rank_map(size,n):
    
    rank_map = []
    for r in range(size):
        iterations = n//size
        i_start = r * iterations
        i_end = i_start + iterations
        if r == size-1:
            i_end += n%size
        rank_map.append({'i_start': i_start, 'i_end': i_end})
    return rank_map
