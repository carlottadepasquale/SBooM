#!/usr/bin/env python3        

#così sa già che deve usare python per eseguirlo, non devo più scrivere python3 genhawkes
from lib.inout import log
import logging
from mpi4py import MPI
import yaml
import os

from lib.inout import console
from lib.inout import reader


comm = MPI.COMM_WORLD #setta il comunicatore: tutti parlano con tutti
size = comm.Get_size() #dice quanti processori stanno lavorando
rank = comm.Get_rank() #assegna il rank
file_param = {}

log.logger_init(rank, "basic")
logger=logging.getLogger('basic')
logger.info('sta funzionando')

if rank==0:
    console_param = console.init()
    path_file_param = os.path.expandvars("$BMCH_HOME") + "/etc/mcconfig.yaml" #expandvars mi ritorna il contenuto della variabile d'ambiente
    file_param = reader.read_yaml_param(path_file_param)
    file_param.update(console_param)
    file_param["size"] = size
    file_param["logger"] = "basic"
    print(file_param)

param = comm.bcast(file_param, root=0)
param["rank"] = rank

if "mc" in param["execution"]:
    from lib.simulator import mcgen
    print(rank, 'Monte Carlo')
    mcgen.simulator(param)
    

    


#get parametri da console 
#get parametri da file
#merge parametri
#montecarlo generator + stima
#output
#plot
#

