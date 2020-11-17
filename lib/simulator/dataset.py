def init_dataset():
    mc_dataset = {}
    mc_dataset['a'] = np.empty(n)
    mc_dataset['b'] = np.empty(n)
    mc_dataset['m'] = np.empty(n)
    mc_dataset['t'] = []
    mc_dataset['n_it'] = 0
    mc_dataset['rank_map'] = [] #i_end e i_start di ogni processore
    return mc_dataset

