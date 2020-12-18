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
    pass

    # logger=logging.getLogger(param["logger"])
    # dir_name = param['dataset_dir'] + param['outprefix'] + "_" + str(param['mu']) + "_" + str(param['alpha']) + "_" + str(param['beta'])+"/"
    # dir_name = dir_name + "N_" + str(param['n']) + "_T_" + str(int(param['t'])) + "/"
    # if param["rank"] == 0:
    #     logger.info(dir_name)


    # count = 0 #i in dataset id va fino a n (id globale), ci serve una variabile che ci dia l'id locale
    # hdf5_file = dir_name + "bootstrap.hdf5"
    # fw = h5py.File(hdf5_file, "w")
    
    # data = np.zeros(param['bt']*dataset['n_local'])
    # data[0:rank+1] = rank
    # data[rank+1:size] = -999
    # print("data",data)
    # f = h5py.File('parallel_test.hdf5', 'w', driver='mpio', comm=comm)
    # dset = []
    # for i in range(size):
    #     #print("i",i)
    #     dset.append(f.create_dataset('test{0}'.format(i), (len(data),), dtype='i'))
    #     dset[i].attrs["rank"] = rank
        
    # print("len dset",rank,len(dset))
    # dset[rank][:] = data  #all2all among task
    # for i in range(size):
    #     for j in range(len(data)):
    #         print(rank,dset[i][j])
    #     print("--")
    # f.close()
    
    
    
    
    
    
    
    # for i in dataset['id']:
    #     t = dataset[''][count]
    #     dset_i = fw.create_dataset("mc_sim_"+str(i), data = t , dtype ='f')
    #     dset_i.attrs['alpha'] = dataset['alpha'][count]
    #     dset_i.attrs['beta'] = dataset['beta'][count]
    #     dset_i.attrs['mu'] = dataset['mu'][count]
    #     dset_i.attrs['stderr'] = dataset['stderr'][count]
    #     count +=1
    
    # fw.close()

    

    