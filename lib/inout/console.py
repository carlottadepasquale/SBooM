import argparse
import random

def init():
    parser = argparse.ArgumentParser(description='Parametri Hawkes e Monte Carlo')
    parser.add_argument('-n','--number_of_interations' , help='Number of iterations of Monte Carlo', required=False) #se metto false posso usare un valore di default
    parser.add_argument('-m','--baseline', help='Initial condition of the Hawkes process')
    parser.add_argument('-a','--intensity', help='alpha, intensity of the process')
    parser.add_argument('-b','--beta', help='beta, decay of the process')
    parser.add_argument('-t','--time', help='time of the process')
    parser.add_argument('--seed', help='seed for the Hawkes process simulator')
    parser.add_argument('-e','--execution', nargs='+', help='type of execution')
    parser.add_argument('--dataset_dir', help='where to save the dataset')
    parser.add_argument('--input_file', help='name of the dataset')
    argn = parser.parse_args()
    console_param = {}
    if argn.number_of_interations:
        n = int(argn.number_of_interations)
        console_param["n"] = n
    if argn.baseline:
        m = float(argn.baseline)
        console_param["mu"] = m
    if argn.intensity:
        a = float(argn.intensity)
        console_param["alpha"] = a
    if argn.beta:
        d = float(argn.beta)
        console_param["beta"] = b
    if argn.time:
        t = float(argn.time)
        console_param["t"] = t
    if argn.seed:
        s = int(argn.seed)
        console_param["seed"] = s
    if argn.execution:
        e = argn.execution
        console_param["execution"] = e
    if argn.dataset_dir:
        dir = argn.dataset_dir
        console_param["dataset_dir"] = dir
    if argn.input_file:
        name = argn.input_file
        console_param["input_file"] = name
    return console_param
