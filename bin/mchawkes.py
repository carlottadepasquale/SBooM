#!/usr/bin/env python3        

#così sa già che deve usare python per eseguirlo, non devo più scrivere python3 genhawkes
from lib.inout import log
import logging

from lib.inout import console


logger=logging.getLogger('basic')
logger.info('sta funzionando')

console_param = console.init()

print(console_param)

#get parametri da console 
#get parametri da file
#merge parametri
#montecarlo generator + stima
#output
#plot
#

