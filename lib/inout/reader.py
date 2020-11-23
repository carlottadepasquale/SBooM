#
# BootMCHawkes
#
# @authors : Carlotta De Pasquale

import yaml
import os
import h5py
from lib.simulator import dataset

def read_yaml_param(path_file):
    """
    reads parameters from yaml file
    input: yaml file path
    output: dictionary
    """
    with open(path_file) as f:
        dict_file_param = yaml.safe_load(f)
    return dict_file_param

def read_hdf5(param):
    hdf5_file = param["dataset_dir"] + param["input_file"]
    if os.path.exists(hdf5_file):
        fr = h5py.File(hdf5_file, 'r')
        param['alpha'] = fr['mc_real_parameters'].attrs['alpha']
        param['beta'] = fr['mc_real_parameters'].attrs['beta']
        param['mu'] = fr['mc_real_parameters'].attrs['mu']
        param['n'] = fr['mc_real_parameters'].attrs['n']
        param['t'] = fr['mc_real_parameters'].attrs['t']

        mc_dataset = dataset.init_dataset(param)

        id_local = 0
        print(fr.keys())
        for id in mc_dataset['id']:
            k = "mc_sim_" + str(id)
            t = (fr[k][:])
            mc_dataset['t'].append(t)
            a=fr[k].attrs['alpha']
            mc_dataset['alpha'][id_local] = a
            b=fr[k].attrs['beta']
            mc_dataset['beta'][id_local] = b
            m=fr[k].attrs['mu']
            mc_dataset['mu'][id_local] = m
            id_local += 1
        return mc_dataset
    else:
        print('ERROR: input_file does not exist')
