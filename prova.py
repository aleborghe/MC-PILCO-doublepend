import pickle as pkl

num_trial = 5
log_file_path = "results_tmp/1/log.pkl"
print("\nLoading policy from: " + log_file_path)
log_dict = pkl.load(open(log_file_path, "rb"))
trial_index = num_trial - 1
parameters_policy = log_dict["parameters_trial_list"][trial_index]