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
    old_size = 0 

    if os.path.exists(json_file):

        if 'bt' in param['execution'] and 'mc' not in param['execution']:
            name_dict = "Montecarlo"
            hdf5_filename_mc = input_dir+"mc_dataset_"
            flag = 'bt'

            param_from_file = json_reader(json_file)
            param['alpha'] = param_from_file[name_dict]['alpha'] 
            param['beta'] = param_from_file[name_dict]['beta'] 
            param['mu'] = param_from_file[name_dict]['mu']    
            param['n'] = param_from_file[name_dict]['n']   
            param['t'] = param_from_file[name_dict]['t']
            old_size = param_from_file[name_dict]['size']

            mcb_dataset = dataset.init_dataset(param)

            rank_mapping(param, mcb_dataset, old_size, hdf5_filename_mc, flag)


        if ('cint1' in param['execution'] or 'cint2' in param['execution'] or \
            'cint3' in param['execution'] or 'cint4' in param['execution'] or \
            'cint5' in param['execution'] ) and 'bt' not in param['execution']:
            name_dict = "Bootstrap_" + str(param['bt'])
            hdf5_filename_mc = input_dir+"mc_dataset_"
            hdf5_filename_bt = input_dir + "bt_" + str(param['bt']) + "/bt_dataset_"
            flag = 'cint'
            
            param_from_file = json_reader(json_file)
            param['alpha'] = param_from_file[name_dict]['alpha'] 
            param['beta'] = param_from_file[name_dict]['beta'] 
            param['mu'] = param_from_file[name_dict]['mu']    
            param['n'] = param_from_file[name_dict]['n']   
            param['t'] = param_from_file[name_dict]['t']
            old_size = param_from_file[name_dict]['size']

            mcb_dataset = dataset.init_dataset(param)

            #first we fills the bootstrap portion of dataset 
            rank_mapping(param, mcb_dataset, old_size, hdf5_filename_bt, flag) 
            
            #we redefine some parameters to then fill the mc portion of dataset
            name_dict = 'Montecarlo'
            old_size = param_from_file[name_dict]['size']
            flag = 'bt'

            rank_mapping(param, mcb_dataset, old_size, hdf5_filename_mc, flag)

    else:
        print('ERROR: input_file does not exist')

    return mcb_dataset

def rank_mapping(param, mcb_dataset, old_size, hdf5_filename, flag):
        
        if param["size"] == old_size: 

            id_local = 0
            
            hdf5_file = hdf5_filename + str(param["rank"])+".hdf5"
            fr = h5py.File(hdf5_file, 'r')
            if flag == 'bt':
                for id in mcb_dataset['id']:
                    fill_dataset_mc(param, mcb_dataset, id, id_local, fr)
                    id_local += 1
        
            if flag == 'cint':
                for id in mcb_dataset['id']:
                    fill_dataset_bt(param, mcb_dataset, id, id_local, fr)
                    id_local += 1
            fr.close()
            

        else:
            
            rank_map_old = dataset.get_rank_map(old_size,param["n"])
            map_files = []
            id_per_file = []
            i_start_new = mcb_dataset["rank_map"][param["rank"]]['i_start'] 
            i_end_new = mcb_dataset["rank_map"][param["rank"]]['i_end']
           
           
            # create map that indicates to each rank which files to read
            file_number = 0
            
            for i in mcb_dataset["id"]:
                for file_number in range(old_size):
                    if  rank_map_old[file_number]["i_start"] <= i < rank_map_old[file_number]["i_end"]:
                        map_files.append(file_number)
                        break
            
            #makes the list elements keys in a dictionary so 
            #they are not repeated and then puts them in another list
            map_files = list(dict.fromkeys(map_files))  #list of files for each rank         

            # for each file in map indicates range of id to be read
            for m in map_files:
                start = 0
                end = 0 

                if i_start_new >= rank_map_old[m]["i_start"]:
                    start = i_start_new
                     
                if i_start_new < rank_map_old[m]["i_start"]: 
                    start = rank_map_old[m]["i_start"]

                if i_end_new >= rank_map_old[m]["i_end"]:
                    end = rank_map_old[m]["i_end"]
                     
                if i_end_new < rank_map_old[m]["i_end"]: 
                    end = i_end_new

                r = range(start,end)
                id_per_file.append(r)

            id_local = 0
            count_idx = 0

            if flag == 'bt': 
                # loop on files
                for f in map_files:
                    hdf5_file = hdf5_filename + str(f)+".hdf5"
                    fr = h5py.File(hdf5_file, 'r')

                    #print(f,count_idx, hdf5_file)
                    
                    for id in id_per_file[count_idx]:
                        fill_dataset_mc(param, mcb_dataset, id, id_local, fr)
                        id_local += 1
                    count_idx += 1
                    fr.close()
            
            if flag == 'cint':
                # loop on files
                for f in map_files:
                    hdf5_file = hdf5_filename + str(f)+".hdf5"
                    fr = h5py.File(hdf5_file, 'r')
                                    
                    for id in id_per_file[count_idx]:
                        fill_dataset_mc(param, mcb_dataset, id, id_local, fr)
                        fill_dataset_bt(param, mcb_dataset, id, id_local, fr)
                        id_local += 1
                    count_idx += 1
                    fr.close()

        #print(param["rank"], mcb_dataset)


def fill_dataset_mc(param, mcb_dataset, id, id_local, fr):
    """
    id = global index number, from 0 to N
    id_local = local rank index number, from 0 to n_local
    fr = hdf5 object file
    
    """

    k = "mc_sim_"+str(id) #nome del dataset hdf5
    t = (fr[k][:])
    mcb_dataset['t'].append(t)
    a=fr[k].attrs['alpha']
    mcb_dataset['alpha'][id_local] = a
    b=fr[k].attrs['beta']
    mcb_dataset['beta'][id_local] = b
    m=fr[k].attrs['mu']
    mcb_dataset['mu'][id_local] = m
    stderr=fr[k].attrs['stderr']
    mcb_dataset['stderr'].append(stderr)
        

def fill_dataset_bt(param, mcb_dataset, id, id_local, fr):
    ba = "bt_alpha_"+str(id) #nome del dataset hdf5
    a = (fr[ba][:])

    bb = "bt_beta_"+str(id)
    b = (fr[bb][:])

    bm = "bt_mu_"+str(id)
    m = (fr[bm][:])

    mcb_dataset['bootstrap'].append({'alpha': a, 'beta': b , 'mu': m})
        