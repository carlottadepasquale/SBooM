#
# BootMCHawkes
#
# @authors : Carlotta De Pasquale
import logging
logger = logging.getLogger('basic')
logger.setLevel( "DEBUG"  )  #prende anche info, warning e critical
ch = logging.StreamHandler() #output a commandline
logger.addHandler(ch)  #aggiunge lo streamhandler al log