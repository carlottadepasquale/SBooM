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

        param_from_file = json_reader(json_file)
        param['alpha'] = param_from_file['alpha'] 
        param['beta'] = param_from_file['beta'] 
        param['mu'] = param_from_file['mu']    
        param['n'] = param_from_file['n']   
        param['t'] = param_from_file['t']
        old_size = param_from_file['size']
        
        mc_dataset = dataset.init_dataset(param)
        
        if param["size"] == old_size: 
            print("OLD == NEW")

            id_local = 0
            
            hdf5_file = input_dir+"mc_dataset_"+str(param["rank"])+".hdf5"
            fr = h5py.File(hdf5_file, 'r')
            for id in mc_dataset['id']:
                k = "mc_sim_"+str(id) #nome del dataset hdf5
                t = (fr[k][:])
                mc_dataset['t'].append(t)
                a=fr[k].attrs['alpha']
                mc_dataset['alpha'][id_local] = a
                b=fr[k].attrs['beta']
                mc_dataset['beta'][id_local] = b
                m=fr[k].attrs['mu']
                mc_dataset['mu'][id_local] = m
                id_local += 1
            print(param["rank"],mc_dataset)

        else:
            
            rank_map_old = dataset.get_rank_map(old_size,param["n"])
            map_files = []
            id_per_file = []
            i_start_new = mc_dataset["rank_map"][param["rank"]]['i_start'] 
            i_end_new = mc_dataset["rank_map"][param["rank"]]['i_end']
           
           
            # create map that indicates to each rank which files to read
            file_number = 0
            
            for i in mc_dataset["id"]:
                for file_number in range(old_size):
                    if  rank_map_old[file_number]["i_start"] <= i < rank_map_old[file_number]["i_end"]:
                        map_files.append(file_number)
                        break

            map_files = list(dict.fromkeys(map_files))

            # for each file in map indicates range of id to be read
            
            #print(param["rank"], map_files) 
            

            for m in map_files:
                start = 0
                end = 0 

                if i_start_new >= rank_map_old[m]["i_start"]:
                    start = i_start_new
                     
                if i_start_new < rank_map_old[m]["i_start"]: 
                    start = rank_map_old[m]["i_start"]

                if i_end_new >= rank_map_old[m]["i_end"]:
                    end = rank_map_old[m]["i_end"]-1
                     
                if i_end_new < rank_map_old[m]["i_end"]: 
                    end = i_end_new-1

                r = range(start,end)
                id_per_file.append(r)

            #print(param["rank"], id_per_file)
            
            
            id_local = 0
            count_idx = 0
            # loop on files
            for f in map_files:

                hdf5_file = input_dir+"mc_dataset_"+str(f)+".hdf5"
                fr = h5py.File(hdf5_file, 'r')

                # loop on index range 
                #for file_range in id_per_file:

                    # loop on id in the range
                for id in id_per_file[count_idx]:

                    k = "mc_sim_"+str(id) #nome del dataset hdf5
                    t = (fr[k][:])
                    mc_dataset['t'].append(t)
                    a=fr[k].attrs['alpha']
                    mc_dataset['alpha'][id_local] = a
                    b=fr[k].attrs['beta']
                    mc_dataset['beta'][id_local] = b
                    m=fr[k].attrs['mu']
                    mc_dataset['mu'][id_local] = m
                    id_local += 1
                
                count_idx =+ 1

        #print(param["rank"], mc_dataset)

        
        return mc_dataset

    else:
        print('ERROR: input_file does not exist')
