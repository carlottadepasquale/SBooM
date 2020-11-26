#
# BootMCHawkes
#
# @authors : Carlotta De Pasquale

import yaml
import os
import h5py
import json
import sys

from lib.simulator import dataset


def json_reader(filepath):
    """
    Checking the existence of json file and read it
    """
    jfc = "-999"
    try:
        if os.path.isfile(filepath) :
            jfc = json.loads(open(filepath).read()) #legge il contenuto del file e lo trasforma in un dizionario
        else :
            sys.exit()

    except:
        sys.exit("ERROR JSON file not readable "+filepath)

    return jfc


def read_yaml_param(path_file):
    """
    reads parameters from yaml file
    input: yaml file path
    output: dictionary
    """
    with open(path_file) as f:
        dict_file_param = yaml.safe_load(f)
    return dict_file_param


def read_dataset(param):
    input_dir = param["dataset_dir"] + param["input"]+"/"
    json_file = input_dir+"param.json"

    if os.path.exists(json_file):
        param_from_file = json_reader(json_file)
        param['alpha'] = param_from_file['alpha'] 
        param['beta'] = param_from_file['beta'] 
        param['mu'] = param_from_file['mu']    
        param['n'] = param_from_file['n']   
        param['t'] = param_from_file['t']

        mc_dataset = dataset.init_dataset(param)

        id_local = 0
        
        for id in mc_dataset['id']:
            hdf5_file = input_dir+"mc_dataset_"+str(id)+".hdf5"
            fr = h5py.File(hdf5_file, 'r')
            k = "mc_sim" #nome del dataset hdf5
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
