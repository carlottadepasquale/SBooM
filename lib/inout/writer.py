#
# BootMCHawkes
#
# @authors : Carlotta De Pasquale

from mpi4py import MPI
import h5py
import numpy as np
import logging
import json
import os

def json_writer(filepath,json_dict):
    """
    Write json file
    """

    with open(filepath, 'w') as json_file:
        json.dump(json_dict, json_file, sort_keys=True, indent=4)

def save_dataset(param, dataset):
    """
    Write the dataset object on file system:

     -param.json : contains param object
     -mc_dataset .hdf5 : contains timeline,alpha,beta,mu for each simulation

    """

    logger=logging.getLogger(param["logger"])
    dir_name = param['dataset_dir'] + param['outprefix'] + "_" + str(param['mu']) + "_" + str(param['alpha']) + "_" + str(param['beta'])+"/"
    dir_name = dir_name + "N_" + str(param['n']) + "_T_" + str(int(param['t'])) + "/"
    if param["rank"] == 0:
        logger.info(dir_name)

    os.makedirs(dir_name,  exist_ok=True) #se esiste già non è un problema ma non sovrascrive
    #write json file with param
    if param["rank"] == 0:
        json_file = dir_name+"param.json"
        json_writer(json_file,param)

    count = 0 #i in dataset id va fino a n (id globale), ci serve una variabile che ci dia l'id locale
    for i in dataset['id']:
        hdf5_file = dir_name+"mc_dataset_"+str(i)+".hdf5"
        fw = h5py.File(hdf5_file, "w")
        t = dataset['t'][count]
        dset_i = fw.create_dataset("mc_sim", data = t , dtype ='f')
        dset_i.attrs['alpha'] = dataset['alpha'][count]
        dset_i.attrs['beta'] = dataset['beta'][count]
        dset_i.attrs['mu'] = dataset['mu'][count]
        count +=1
    
    fw.close()

    

    