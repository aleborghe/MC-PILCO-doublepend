import copy
import pickle as pkl
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import signal

import gpr_lib.Likelihood.Gaussian_likelihood as Likelihood
import gpr_lib.Utils.Parameters_covariance_functions as cov_func
import model_learning.Model_learning as ML
import policy_learning.Cost_function as Cost_function
import policy_learning.MC_PILCO as MC_PILCO
import policy_learning.Policy as Policy
import simulation_class.ode_systems as f_ode

# define paths
seed = 2
folder_path = "results_tmp/" + str(seed) + "/"
config_file_path = folder_path + "/config_log.pkl"
saving_seed = 20

# number of simulated particles
num_particles = 50

# select the policy obtained at trial 'num_trial'
num_trial = 5
past_trials = 10

# initialize the object
config_dict = pkl.load(open(config_file_path, "rb"))
config_dict["reinforce_param_dict"]["loaded_model"] = True
config_dict["reinforce_param_dict"]["num_trials"] = num_trial
PL_obj = MC_PILCO.MC_PILCO(**config_dict["MC_PILCO_init_dict"])

# load policy
PL_obj.load_policy_from_log(past_trials, folder=folder_path)

# load model
PL_obj.load_model_from_log(past_trials, folder=folder_path)

PL_obj.reinforce(**config_dict["reinforce_param_dict"])


pkl.dump(config_log_dict, open("results_tmp/" + str(saving_seed) + "/config_log.pkl", "wb"))
