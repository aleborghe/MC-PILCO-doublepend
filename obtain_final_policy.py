import argparse
import pickle as pkl
import numpy as np

# file parameters
p = argparse.ArgumentParser("plot log")
p.add_argument("-dir_path", type=str, default="results_tmp/", help="none")
p.add_argument("-seed", type=int, default=1, help="none")

# load parameters
locals().update(vars(p.parse_known_args()[0]))
file_name = dir_path + str(seed) + "/log.pkl"
print("---- Reading log file: " + file_name)
log_dict = pkl.load(open(file_name, "rb"))
particles_states_list = log_dict["particles_states_list"]
num_trials = len(particles_states_list)

print("\nLoading policy from: " + file_name)
log_dict = pkl.load(open(file_name, "rb"))
trial_index = num_trials-1
parameters_policy = log_dict["parameters_trial_list"][trial_index]
policy_par = {}
policy_par["log_lengthscales"] = parameters_policy["log_lengthscales"].cpu().detach().numpy()
policy_par["centers"] = parameters_policy["centers"].cpu().detach().numpy()
policy_par["linear_weights"] = parameters_policy["f_linear.weight"].cpu().detach().numpy()
print(policy_par)
