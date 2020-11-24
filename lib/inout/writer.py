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
    fw = h5py.File(filename, "w", driver='mpio', comm=MPI.COMM_WORLD)

    group = fw.create_group("mc_real_parameters")
    group.attrs['n'] = param['n']
    group.attrs['t'] = param['t']
    group.attrs['alpha'] = param['alpha']
    group.attrs['beta'] = param['beta']
    group.attrs['mu'] = param['mu']

    dset_a = fw.create_dataset("alpha" , (param['n'],), dtype ='f')
    dset_b = fw.create_dataset("beta" , (param['n'],), dtype ='f')
    dset_m = fw.create_dataset("mu" , (param['n'],), dtype ='f')
    
    dset_a[dataset['id']] = dataset['alpha']
    dset_b[dataset['id']] = dataset['beta']
    dset_m[dataset['id']] = dataset['mu']
    # dset_list = []
    # for i in range(len(dataset['t'])):
    #     # print("dataset[t]: ", type(dataset['t'][i]), dataset['t'][i].shape)
    #     dset_list.append(fw.create_dataset("t"+str(i) , (dataset['t'][i].shape[0],), dtype ='f'))
    #     print("rank n ", param['rank'], ", dset_list: ", len(dset_list))
        
        
    # # print(len(dset_list))
    
    # # print(dataset['id'])
    # #print(len(dataset['t']))
    # i_loc = 0
    # logger.info(str(dataset['id'])+" "+str(len(dset_list)))
    # for i in dataset['id']:

    #     print(str(i)+" "+str(i_loc))
    #     # print(type(dset_list[i]))
    #     # print(dset_list[i].shape)
    #     # print(type(dataset['t'][i_loc]))
    #     # print(dataset['t'][i_loc].shape)
    #     # print(dataset['t'][i_loc])
    #     logger.info("lenght: "+ str(len(dset_list)) + " " + str(len(dataset['t'])))
    #     dset_list[i][:] = dataset['t'][i_loc]
    #     i_loc = i_loc + 1

    # # print(type(dataset['t'][0]))
    # # print(type(dataset['t'][0][0]))
    # # print(type(dataset['t'][0:dataset['n_local']][:]))
      
        
    # # print("dataset[t]: ", type(dataset['t']), dataset['t'][i].shape)
    # # print("dset_t: ", type(dset_t))
    # # # for i in dataset['id']:
    # #     for d in range(len(dataset['t'][i])):
    # #         dset_t[i][d] = dataset['t'][d]
        

    
    
    # # count = 0 #i in dataset id va fino a n (id globale), ci serve una variabile che ci dia l'id locale
    # # for i in dataset['id']:
    # #     t = dataset['t'][count]
    # #     dset_i = fw.create_dataset("mc_sim_"+ str(i), data = t , dtype ='i')
    # #     dset_i.attrs['alpha'] = dataset['alpha'][count]
    # #     dset_i.attrs['beta'] = dataset['beta'][count]
    # #     dset_i.attrs['mu'] = dataset['mu'][count]
    # #     count +=1
    
    fw.close()
    