import argparse


def init():
    parser = argparse.ArgumentParser(description='Parametri Hawkes e Monte Carlo')
    parser.add_argument('-n','--number_of_interations' , help='Number of iterations of Monte Carlo', required=False) #se metto false posso usare un valore di default
    parser.add_argument('-m','--baseline', help='Initial condition of the Hawkes process')
    parser.add_argument('-a','--intensity', help='alpha, intensity of the process')
    parser.add_argument('-d','--decay', help='delta, decay of the process')
    parser.add_argument('-t','--time', help='time of the process')
    argn = parser.parse_args()
    console_param = {}
    if argn.number_of_interations:
        n = int(argn.number_of_interations)
        console_param["n"] = n
    if argn.baseline:
        m = float(argn.baseline)
        console_param["m"] = m
    if argn.intensity:
        a = float(argn.intensity)
        console_param["a"] = a
    if argn.decay:
        d = float(argn.decay)
        console_param["d"] = d
    if argn.time:
        t = float(argn.time)
        console_param["t"] = t
    return console_param
