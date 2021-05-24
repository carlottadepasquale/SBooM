SBooM: Simulator of Bootstrap  and Monte Carlo for Hawkes processes
=================================
**SBooM is a parallel software for HPC written to carry out the confidence intervals calculations of the estimates Hawkes processes parameters using the bootstrap method.**


First, it simulates *N* Hawkes processes with exponential kernel using a set of predefined parameters. Subsequently for each simulated process it estimates the parameters of the intensity function α, β and μ and their standard errors by using the Hessian matrix as an approximation for the variance-covariance matrix. They are used to compute the asymptotic 95% confidence intervals.

Afterwards it executes *B* bootstrap resamples for each one of the sets of estimated parameters. Once the estimates of the parameters of the B resamples are completed, they are used to calculate the confidence intervals of the estimated parameters using different methods.

Lastly, the accuracy of the bootstrap confidence intervals was compared with the accuracy of the asymptotic confidence intervals.

Theoretical Framework
----------------------
**Hawkes processes** are a particular type of self-exciting point processes where the happening of an event increases the probability of another happening in the future. Some examples of events that can be modeled through Hawkes processes are stock market sales, social media posts virality or stimulaitons between neurons. Their function is composed of a baseline parameter, indicated with μ, that represents the exogenous rate of arrival of events in the process, and of a non-linear memory kernel, that makes the estimation of parameters extremely computationally intensive. 

**Monte Carlo simulations** are applicable when dealing with questions that cannot be answered though asymptotic theory. Even if there are several different types of Monte Carlo algorithm, they are all based on the random generation of a stochastic process, under a set of controlled conditions, for a substantial number of times. These repetitions are used to artificially create the sampling distribution of an estimator and derive its properties.

**The bootstrap method** is a resampling technique that is used to derive the asymptotic distribution of an estimator by sampling from a given dataset with replacement.

Both the Monte Carlo and the bootstrap are computationally intensive methods, and including them in an experiment might become infeasible if applied to the calculation of a complex statistic such as parameter estimation in Hawkes processes, at least not with the necessary number of iterations needed for the experiment to be reliable. 

This is why we developed a parallel software to perform this experiment.
 

Setup and configuration
------------------------

### Python packages required
In order to run the software, there are several python packages that need to be installed:
- numpy
- mpi4py
- h5py
- scipy
- pyyaml
- cython
- matplotlib
- pandas
- seaborn

### Run parameters
In the **etc/default** folder there is a file called *mcconfig.yaml* containing standard run parameters. In order to be able to easily switch from one computer to another without having to commit, the user can create a file with the same name in the folder above. The parameters inserted in this file if present will overwrite the ones in **etc/default**.
Run parameters can also be given through the commandline.
Run parameters are:
- the parameters of the Hawkes process:
    - α **(flag -a)**,
    - β **(flag -b)**,
    - μ **(flag -m)**,
- length of the time interval considered *T* **(flag -t)**
- number of Monte Carlo iterations *N* **(flag -n)**
- number of bootstrap repetitions *B* **(flag -bt)**
- seed to have the same results in different runs **(flag --seed)**

A peculiarity of this software is that the user can choose which parts of the program he wants to run using the **flag -e**. The options are:
- **mc**: performs the parallel simulation and estimation of the N Hawkes processes.
- **save_mc**: saves the result of the Monte Carlo simulation on the computer in a preset folder.
- **bt**: performs in parallel the B simulations over the parameter estimated in MC and the estimation of their parameters.
- **save_bt**: saves the result found by B in a preset folder.
- **cint1-cint3**: perform the confidence intervals of the bootstrap estimates using 3 different methods.
- **plt**: plots graphs of the simulated process
- **plt_cint**: plots graphs about the confidence intervals

The software can also perform the bootstrap simulation or the confidence intervals calculation starting from a previously simulated dataset. By using the flag **--input** the user can give the software the path of a previously saved simulation on which to compute the desired steps.

In case the user does not want to simulate the process since he or she already has a Hawkes process on which parameters and confidence intervals have to be estimated, the code can be launched with the flags **-e analyze --input csv_path** (where csv_path indicates the parth of the csv file containing the process). This way α, β and μ will be estimated. By adding **bt** to the -e flag, the code will also perform the estimtion of bootstrap confidence intervals.
### Setup
In the **bin** folder there is a *setup.sh* file that sets environment variables. It can be activated through the source command in commandline.

Run examples
------------

**srun -n 4 sboom.py -e mc save_mc -n 1000 -t 50 -a 0.5 -b 5 -m 0.5**

This run performs the simulation and estimation of N=1000 Hawkes processes with α=0.5, β=5 and μ=0.5. If the estimated paraeters do not respect the conditions for stationarity, or if there are errors in the calculation of the standard error, the process is discarded and re-simulated. After the parameters are estimated for N processes, the standard errors are used to calculate asymptotic standard errors. The number of confidence intervals containing the real value of the parameter are counted and reported in percentage.

In the output we can also see the percentage of discarded sample and the reason why they were discarded (**alpha/beta/mu out of bound**, **stderr calculation fails**), the time needed to perform the individual parts (in this case, **Monte Carlo Time** and **save time dataset**)of the code and the overall time to solution, as well as a list of all the run parameters used.

The line with **save_dataset** indicates the path of the folder where the Monte Carlo simulaiton is saved. This path can later be used with the flag **--input** to perform the bootstrap without redoing the Monte Carlo.

    rank: 0 13:34 [CRITICAL] (main) : START Mchawkes.py
    rank: 0 13:34 [CRITICAL] (main) : Reader time: 0.000 s
    rank: 0 13:34 [INFO] (simulator) : 
    - mu_asymptotic:    93.8%
    - alpha_asymptotic: 92.5%
    - beta_asymptotic:  93.6%

    - alpha* out of bound:  0.0%
    - beta out of bound:  0.0%
    - mu out of bound:  0.1%
    - alpha (br) out of bound:  0.1%
    - Stderr calculation fails: 0.0
    - discarded samples:  0.2%

    - Theoretical avg n of events: 50.0
    - Avg n of events: 49.7
    - Avg time Hawkes: 0.00337
    - Avg time inference: 0.02632
    rank: 0 13:34 [CRITICAL] (main) : Montecarlo time: 8.820 s
    rank: 0 13:34 [INFO] (save_dataset) : xx/Dataset_hdf5/mc_dataset_0.5_0.5_5.0/N_1000_T_50/
    rank: 0 13:34 [CRITICAL] (main) : Save time dataset: 0.161 s
    rank: 0 13:34 [CRITICAL] (main) : Bootstrap time: 0.000 s
    rank: 0 13:34 [CRITICAL] (main) : Save time bootstrap: 0.000 s
    rank: 0 13:34 [CRITICAL] (main) : Confidence intervals time: 0.000 s
    rank: 0 13:34 [CRITICAL] (main) : RUN PARAMETERS:
    MPI Size: 4
    Number of iterations: 1000
    Hawkes T: 50.0
    Number of bootstrap it: 6
    alpha, beta, mu: 0.5   5.0   0.5

    rank: 0 13:34 [CRITICAL] (main) : Time to solution: 8.984 s
    rank: 0 13:34 [CRITICAL] (main) : END Mchawkes.py


**srun sboom.py -e mc bt save_bt cint1 cint2 -n 100000 -t 50 -bt 399 --seed 1234**

In this case we are performing the Monte Carlo and the bootstrap simulations, as well as the confidence intervals using methods 1 and 2. In the output we can see the percentages of times where the bootstrap confidence intervals contain the real parameter value for each of the methods used (**alpha/beta/mu_bt_method1/2**)

    rank: 0 17:26 [CRITICAL] (main) : START Mchawkes.py
    rank: 0 17:26 [CRITICAL] (main) : Reader time: 0.000 s
    rank: 0 17:30 [INFO] (simulator) : 
    - mu_asymptotic:    94.77%
    - alpha_asymptotic: 93.67%
    - beta_asymptotic:  94.87%

    - alpha* out of bound:  0.0%
    - beta out of bound:  0.0%
    - mu out of bound:  0.0%
    - alpha (br) out of bound:  0.0%
    - Stderr calculation fails: 0.0
    - discarded samples:  0.0%

    - Theoretical avg n of events: 100.0
    - Avg n of events: 100.1
    - Avg time Hawkes: 0.01384
    - Avg time inference: 0.11642
    rank: 0 17:30 [CRITICAL] (main) : Montecarlo time: 187.198 s
    rank: 0 17:30 [INFO] (save_dataset) : /xxx/hdf5_dataset/mc_dataset_0.5_0.5_25.0/N_10000_T_100/
    rank: 0 17:30 [CRITICAL] (main) : Save time dataset: 0.047 s

    rank: 0 20:07 [CRITICAL] (main) : Bootstrap time: 9474.459 s
    rank: 0 20:07 [INFO] (save_bootstrap) : /xxx/hdf5_dataset/mc_dataset_0.5_0.5_25.0/N_10000_T_100/
    rank: 0 20:08 [CRITICAL] (main) : Save time bootstrap: 1.521 s
    rank: 0 20:08 [INFO] (confidence_int_1) : 
    - method1_alpha_bt: 92.940%
    - method1_beta_bt:   95.790%
    - method1_mu_bt:     94.450%

    rank: 0 20:08 [INFO] (confidence_int_2) : 
    - method2_alpha_bt: 91.640%
    - method2_beta_bt: 93.270%
    - method2_mu_bt: 94.090%

    rank: 0 20:08 [CRITICAL] (main) : Confidence intervals time: 0.167 s
    rank: 0 20:08 [CRITICAL] (main) : RUN PARAMETERS:
    MPI Size: 360
    Number of iterations: 10000
    Hawkes T: 100.0
    Number of bootstrap it: 299
    alpha, beta, mu: 0.5   25.0   0.5

    rank: 0 20:08 [CRITICAL] (main) : Time to solution: 9664.111 s
    rank: 0 20:08 [CRITICAL] (main) : END Mchawkes.py

**sboom.py -e analyze --input_csv /xx/xxx/all.csv -bt 199**

In this run we are performing the analysis of an external dataset, "all.csv". First, in the output we can see the estimated values of the parameters (**alpha/beta/mu est**), as well as the bootstrap confidence intervals performed according to several different methods.

    rank: 0 12:11 [CRITICAL] (main) : START Mchawkes.py
    t:  20
    alpha est:  0.9867361123628087
    beta est:  8.224401810906734
    mu est:  1.187117976630074
    rank: 0 12:21 [INFO] (confidence_int_1) : 
    -alpha confidence interval 1: 0.7232603990023615 , 0.9979734075644123
    -beta confidence interval 1: 5.63052171443813 , 11.405516034841341
    -mu confidence interval 1: 0.5127691825541639 , 0.9925622364578877

    rank: 0 12:21 [INFO] (confidence_int_2) : 
    -alpha confidence interval 2: 1.2502118257232557 , 0.975498817161205
    -beta confidence interval 2: 10.818281907375338 , 5.043287586972127
    -mu confidence interval 2: 1.861466770705984 , 1.3816737168022601

    rank: 0 12:21 [INFO] (confidence_int_3) : 
    -alpha confidence interval 3: 0.9765629135558964 , 0.9969093111697209
    -beta confidence interval 3: 8.015881400109922 , 8.432922221703546
    -mu confidence interval 3: 1.1683352221195076 , 1.2059007311406402

    rank: 0 12:21 [INFO] (confidence_int_4) : 
    -alpha confidence interval 4: 1.2502118257232557 , 0.975498817161205
    -beta confidence interval 4: 10.818281907375338 , 5.043287586972127
    -mu confidence interval 4: 1.861466770705984 , 1.3816737168022601

    rank: 0 12:21 [INFO] (confidence_int_5) :
    -alpha confidence interval 5: 0.9762811088704298 , 0.9971911158551875
    -beta confidence interval 5: 3.8319575148167084 , 12.61684610699676
    -mu confidence interval 5: 1.1514787982286292 , 1.2227571550315186

    rank: 0 12:21 [CRITICAL] (main) : Reader time: 0.000 s
    rank: 0 12:21 [CRITICAL] (main) : Montecarlo time: 0.000 s
    rank: 0 12:21 [CRITICAL] (main) : Save time dataset: 0.000 s
    rank: 0 12:21 [CRITICAL] (main) : Bootstrap time: 0.000 s
    rank: 0 12:21 [CRITICAL] (main) : Save time bootstrap: 0.000 s
    rank: 0 12:21 [CRITICAL] (main) : Confidence intervals time: 0.000 s
    rank: 0 12:21 [CRITICAL] (main) : RUN PARAMETERS:
    MPI Size: 1
    Number of iterations: 1
    Hawkes T: 20
    Number of bootstrap it: 199
    alpha, beta, mu: 0.8   5   0.5

    rank: 0 12:21 [CRITICAL] (main) : Time to solution: 643.880 s
    rank: 0 12:21 [CRITICAL] (main) : END Mchawkes.py
