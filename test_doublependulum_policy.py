import os
from datetime import datetime
import argparse
import pickle as pkl

import matplotlib.pyplot
import numpy as np


from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.controller.combined_controller import CombinedController
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.controller.mcpilco.mcpilco_controller import Controller_sum_of_Gaussians_with_angles_numpy as MCPilcoController

from double_pendulum.analysis.leaderboard import get_swingup_time


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




def condition1(t, x):
    return False

def condition2(t, x):
    if np.allclose(x[:2], [np.pi, 0.0], atol=1e-2):
        return True
    return False

model_par_path = 'pendubot_parameters.yml'
torque_limit = [6.0, 0.0]
active_act = 0

mpar = model_parameters(filepath=model_par_path)
mpar.set_torque_limit(torque_limit)

dt = 0.001
t_final = 10.0
integrator = "runge_kutta"
goal = [np.pi, 0.0, 0.0, 0.0]
plant = SymbolicDoublePendulum(model_pars=mpar)
sim = Simulator(plant=plant)

controller1 = MCPilcoController(parameters=policy_par, ctrl_rate=5, u_max=torque_limit[active_act], controlled_dof = 0, wait_steps = 0)

controller2 = LQRController(model_pars=mpar)
controller2.set_goal(goal)
Q = 3.0 * np.diag([0.5, 0.5, 0.1, 0.1])
R = np.eye(2) * 0.1
controller2.set_cost_matrices(Q=Q, R=R)
controller2.set_parameters(failure_value=0.0,
                          cost_to_go_cut=15)

# initialize combined controller
controller = CombinedController(
    controller1=controller1,
    controller2=controller2,
    condition1=condition1,
    condition2=condition2,
    compute_both=False
)

#controller = controller1
controller.init()

# start simulation
T, X, U = sim.simulate_and_animate(
    t0=0.0,
    x0=[0.0, 0.0, 0.0, 0.0],
    tf=t_final,
    dt=dt,
    controller=controller,
    integrator=integrator,
    save_video=False,
)

print('Swingup time: '+str(get_swingup_time(T, np.array(X), mpar=mpar)))


# plot timeseries
plot_timeseries(
    T,
    X,
    U,
    X_meas=sim.meas_x_values,
    pos_y_lines=[np.pi],
    tau_y_lines=[-torque_limit[active_act], torque_limit[active_act]],
)