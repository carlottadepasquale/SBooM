#!/usr/bin/env python3        

#così sa già che deve usare python per eseguirlo, non devo più scrivere python3 genhawkes
from lib.inout import log
import logging
from mpi4py import MPI
import yaml
import os

from lib.inout import console
from lib.inout import reader
from lib.simulator import dataset


comm = MPI.COMM_WORLD #setta il comunicatore: tutti parlano con tutti
size = comm.Get_size() #dice quanti processori stanno lavorando
rank = comm.Get_rank() #assegna il rank
file_param = {}

log.logger_init(rank, "basic")
logger=logging.getLogger('basic')

if rank==0:
    console_param = console.init()
    path_file_param = os.path.expandvars("$BMCH_HOME") + "/etc/mcconfig.yaml" #expandvars mi ritorna il contenuto della variabile d'ambiente
    file_param = reader.read_yaml_param(path_file_param)
    file_param.update(console_param)
    file_param["size"] = size
    file_param["logger"] = "basic"
    
param = comm.bcast(file_param, root=0)
param["rank"] = rank

dataset = dataset.init_dataset(param)

if "mc" in param["execution"]:
    from lib.simulator import mcgen
    logger.info('Monte Carlo')
    mcgen.simulator(param, dataset)


if "save" in param["execution"]:
    from lib.inout import writer
    comm.Barrier()
    writer.save_hdf5(param, dataset)



    


#get parametri da console 
#get parametri da file
#merge parametri
#montecarlo generator + stima
#output
#plot
#

