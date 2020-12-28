#
# BootMCHawkes
#
# @authors : Carlotta De Pasquale
import logging

def logger_init(rank, logger_name, logger_level):
    logger = logging.getLogger(logger_name)
    logger.setLevel( logger_level  )  #DEBUG: prende anche info, warning e critical
    ch = logging.StreamHandler() #output a commandline
    formatter = logging.Formatter("rank: " + str(rank)+ ' %(asctime)s [%(levelname)s] (%(funcName)s) : %(message)s',"%H:%M")
    ch.setFormatter(formatter)
    logger.addHandler(ch)  #aggiunge lo streamhandler al log