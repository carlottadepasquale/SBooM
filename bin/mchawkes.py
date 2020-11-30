#!/usr/bin/env python3        

#così sa già che deve usare python per eseguirlo, non devo più scrivere python3 genhawkes
from lib.inout import log
import logging
from mpi4py import MPI
import yaml
import os
import time

from lib.inout import console
from lib.inout import reader
from lib.simulator import dataset


comm = MPI.COMM_WORLD #setta il comunicatore: tutti parlano con tutti
size = comm.Get_size() #dice quanti processori stanno lavorando
rank = comm.Get_rank() #assegna il rank
file_param = {}

tts0 = MPI.Wtime()

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

tread0 = MPI.Wtime()
if ("bt" in param["execution"] or "plt" in param["execution"]) and "input" in param:
    dataset = reader.read_dataset(param)

else:
    dataset = dataset.init_dataset(param)
tread = MPI.Wtime() - tread0

tmc0 = MPI.Wtime()
if "mc" in param["execution"]:
    from lib.simulator import mcgen
    mcgen.simulator(param, dataset, comm)
tmc = MPI.Wtime() - tmc0

tsave0 = MPI.Wtime()
if "save" in param["execution"]:
    from lib.inout import writer
    comm.Barrier()
    writer.save_dataset(param, dataset)
tsave = MPI.Wtime() - tsave0

if "plt" in param["execution"]:
    from lib.inout import plot
    plot.plot_estimate(param, dataset, comm)

tts = MPI.Wtime() - tts0

if rank==0:
    logger.critical("MPI Size: "+ str(size))
    logger.critical("Number of iterations: "+ str(param["n"]))
    logger.critical("Hawkes T: "+ str(param["t"]))
    logger.critical("Time to solution: "+ str(tts)) 
    logger.critical("Reader time: "+ str(tread)) 
    logger.critical("Montecarlo time: "+ str(tmc)) 
    logger.critical("Save time: "+ str(tsave)) 






#get parametri da console 
#get parametri da file
#merge parametri
#montecarlo generator + stima
#output
#plot
#

