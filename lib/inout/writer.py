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

from lib.inout import reader

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
        dict_param = {'Montecarlo': param}
        json_writer(json_file,dict_param)

    count = 0 #i in dataset id va fino a n (id globale), ci serve una variabile che ci dia l'id locale
    hdf5_file = dir_name+"mc_dataset_"+str(param["rank"])+".hdf5"
    fw = h5py.File(hdf5_file, "w")
    for i in dataset['id']:
        t = dataset['t'][count]
        dset_i = fw.create_dataset("mc_sim_"+str(i), data = t , dtype ='f')
        dset_i.attrs['alpha'] = dataset['alpha'][count]
        dset_i.attrs['beta'] = dataset['beta'][count]
        dset_i.attrs['mu'] = dataset['mu'][count]
        dset_i.attrs['stderr'] = dataset['stderr'][count]
        count +=1
    
    fw.close()

def save_bootstrap(param, dataset):
    """
    Write the dataset object on file system:

     -param.json : contains param object
     -mc_dataset .hdf5 : contains timeline,alpha,beta,mu for each simulation

    """
    logger=logging.getLogger(param["logger"])
    dir_name = param['dataset_dir'] + param['outprefix'] + "_" + str(param['mu']) + "_" + str(param['alpha']) + "_" + str(param['beta'])+"/"
    dir_name = dir_name + "N_" + str(param['n']) + "_T_" + str(int(param['t'])) + "/"
    dir_name_hdf5 = dir_name+ "bt_" + str(param['bt'])+ "/"
    if param["rank"] == 0:
        logger.info(dir_name)

    os.makedirs(dir_name_hdf5,  exist_ok=True) #se esiste già non è un problema ma non sovrascrive
    #write json file with param
    
    json_file = dir_name+"param.json"
    
    if param["rank"] == 0:
        dict_param = {}
        if os.path.exists(json_file):
            dict_param = reader.json_reader(json_file)
        dict_param['Bootstrap_' + str(param['bt'])] = param 
        param['bootstrap_dataset_dir'] = dir_name_hdf5
        json_writer(json_file,dict_param)

    count = 0 #i in dataset id va fino a n (id globale), ci serve una variabile che ci dia l'id locale
    hdf5_file = dir_name_hdf5 + "bt_dataset_"+str(param["rank"])+".hdf5"
    fw = h5py.File(hdf5_file, "w")
    for i in dataset['id']:
        dset_a_i = fw.create_dataset("bt_alpha_"+str(i), data = dataset['bootstrap'][count]['alpha'] , dtype ='f')
        dset_b_i = fw.create_dataset("bt_beta_"+str(i), data = dataset['bootstrap'][count]['beta'] , dtype ='f')
        dset_m_i = fw.create_dataset("bt_mu_"+str(i), data = dataset['bootstrap'][count]['mu'] , dtype ='f')
        count +=1
    
    fw.close()
 