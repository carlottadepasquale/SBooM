#!/usr/bin/env python3        

#
# SBooM
#
# @authors : Carlotta De Pasquale

def main():

    import logging
    from mpi4py import MPI
    import yaml
    import os
    import time

    from lib.inout import log
    from lib.inout import console
    from lib.inout import reader
    from lib.simulator import dataset

    comm = MPI.COMM_WORLD #setta il comunicatore: tutti parlano con tutti
    size = comm.Get_size() #dice quanti processori stanno lavorando
    rank = comm.Get_rank() #assegna il rank
    file_param = {}

    tts0 = MPI.Wtime()


    time_format = "{t:.3f} s"
    if rank==0:
        console_param = console.init()
        path_file_param_default = os.path.expandvars("$BMCH_HOME") + "/etc/default/mcconfig.yaml" #expandvars mi ritorna il contenuto della variabile d'ambiente
        path_file_param_sec = os.path.expandvars("$BMCH_HOME") + "/etc/mcconfig.yaml"
        file_param = reader.read_yaml_param(path_file_param_default)
        file_param_sec = reader.read_yaml_param(path_file_param_sec)
        file_param.update(file_param_sec)
        file_param.update(console_param)
        file_param["size"] = size
        file_param["logger"] = "basic"

    param = comm.bcast(file_param, root=0)
    log.logger_init(rank, param["logger"], param["log_level"])
    logger=logging.getLogger('basic')
    if rank==0:
        logger.critical("START Mchawkes.py")
    param["rank"] = rank

    ####################

    tread0 = MPI.Wtime()
    if "input" in param:
        dataset = reader.read_dataset(param)

    else:
        dataset = dataset.init_dataset(param)
    tread = MPI.Wtime() - tread0

    #####################

    if "analyze" in param['execution']:
        reader.read_csv(param, dataset)
        from lib.analyzer import estimate_csv
        estimate_csv.estimate_exp(param, dataset)
        estimate_csv.bootstrap_cint(param, dataset, comm)

    #####################

    if rank==0:    
        logger.critical("Reader time: "+ time_format.format(t=tread))

    ####################

    tmc0 = MPI.Wtime()
    if "mc" in param["execution"]:
        from lib.simulator import mcgen
        mcgen.simulator(param, dataset, comm)
    tmc = MPI.Wtime() - tmc0

    if rank==0:
        logger.critical("Montecarlo time: "+ time_format.format(t=tmc)) 

    ####################

    tsave0 = MPI.Wtime()
    if "save_mc" in param["execution"]:
        from lib.inout import writer
        #comm.Barrier()
        writer.save_dataset(param, dataset)
    tsave_dset = MPI.Wtime() - tsave0

    if rank==0:
        logger.critical("Save time dataset: "+ time_format.format(t=tsave_dset)) 

    ####################

    tbt0 = MPI.Wtime()
    if "bt" in param["execution"]:
        from lib.simulator import bootstrap
        bootstrap.bootstrap(param, dataset, comm)
        time_barr_bt0= MPI.Wtime()
        comm.Barrier()
        time_barr_bt= MPI.Wtime()-time_barr_bt0
        logger.critical(str(rank) +"   "+ time_format.format(t=time_barr_bt))
    tbt = MPI.Wtime() - tbt0

    if rank==0:
        logger.critical("Bootstrap time: "+ time_format.format(t=tbt)) 

    ####################

    tsave_bt0 = MPI.Wtime()
    if "save_bt" in param["execution"]:
        from lib.inout import writer
        writer.save_bootstrap(param, dataset)
    tsave_bt = MPI.Wtime() - tsave_bt0

    if rank==0:
        logger.critical("Save time bootstrap: "+ time_format.format(t=tsave_bt)) 

    ####################

    if "plt" in param["execution"]:
        from lib.inout import plot
        plot.plot_estimate(param, dataset, comm)

    ####################

    if "plt_pr" in param["execution"]:
        from lib.inout import plot
        plot.plot_process(dataset)

    ####################

    tcint0 = MPI.Wtime()

    if "cint1" in param["execution"]:
        from lib.simulator import bootstrap
        bootstrap.confidence_int_1(param, dataset, comm)

    if "cint2" in param["execution"]:
        from lib.simulator import bootstrap
        bootstrap.confidence_int_2(param, dataset, comm)

    if "cint3" in param["execution"]:
        from lib.simulator import bootstrap
        bootstrap.confidence_int_3(param, dataset, comm)

    if "cint4" in param["execution"]:
        from lib.simulator import bootstrap
        bootstrap.confidence_int_4(param, dataset, comm)

    if "cint5" in param["execution"]:
        from lib.simulator import bootstrap
        bootstrap.confidence_int_5(param, dataset, comm)

    tcint = MPI.Wtime() - tcint0

    if rank==0:
        logger.critical("Confidence intervals time: "+ time_format.format(t=tcint)) 

    ####################
    if "plt_cint" in param["execution"]:
        from lib.inout import plot
        plot.plot_cint(param, dataset, comm)
    ####################

    tts = MPI.Wtime() - tts0

    if rank==0:
        log_str = "RUN PARAMETERS:"
        log_str += ('\n' +"MPI Size: "+ str(size) + '\n')
        log_str += ("Number of iterations: "+ str(param["n"]) + '\n')
        log_str += ("Hawkes T: "+ str(param["t"]) + '\n')
        log_str += ("Number of bootstrap it: "+ str(param["bt"]) + '\n')
        log_str += ("alpha, beta, mu: "+ str(param["alpha"]) +"   "+ str(param["beta"])+"   " + str(param["mu"]) +'\n')
        logger.critical(log_str)
        logger.critical("Time to solution: "+ time_format.format(t=tts) ) 
        logger.critical("END Mchawkes.py")

#se questo script è lanciato come main (mpirun nomescript.py) esegie la funzione name
if __name__ == "__main__": 
    main()





#get parametri da console 
#get parametri da file
#merge parametri
#montecarlo generator + stima
#output
#plot
#

