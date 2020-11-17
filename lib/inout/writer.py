#
# BootMCHawkes
#
# @authors : Carlotta De Pasquale

from mpi4py import MPI
import h5py
import numpy as np
import logging

#write file
def save_hdf5(param, dataset):
    logger=logging.getLogger(param["logger"])
    filename = param['dataset_dir'] + param['outprefix'] + "_" + str(param['n']) + "_" + str(param['t']) + ".hdf5"
    logger.info(filename)
    fw = h5py.File(filename, "w", driver='mpio')
    group = fw.create_group("mc_real_parameters")
    group.attrs['n'] = param['n']
    group.attrs['t'] = param['t']
    group.attrs['alpha'] = param['alpha']
    group.attrs['beta'] = param['beta']
    group.attrs['mu'] = param['mu']

    for i in dataset['id']:
        t = dataset['t'][i]
        dset_i = fw.create_dataset("mc_sim_"+ str(i), data = t , dtype ='i')
        dset_i.attrs['alpha'] = dataset['alpha'][i]
        dset_i.attrs['beta'] = dataset['beta'][i]
        dset_i.attrs['mu'] = dataset['mu'][i]
    
    fw.close()
    