# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Authors: 	Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
         	Fabio Amadio (fabioamadio93@gmail.com)
MERL contact:	Diego Romeres (romeres@merl.com)
"""

import pickle as pkl

import numpy as np
import sympy as sym


def pend(y, t, u):
    """
    System of first order equations for a pendulum system
    The policy commands the torque applied to the joint
    (stable equilibrium point with the pole down at [0,0])
    """
    theta, theta_dot = y

    m = 1.0  # mass of the pendulum
    l = 1.0  # lenght of the pendulum
    b = 0.1  # friction coefficient
    g = 9.81  # acceleration of gravity
    I = 1 / 3 * m * l**2  # moment of inertia of a pendulum around extreme point

    dydt = [theta_dot, (u - b * theta_dot - 1 / 2 * m * l * g * np.sin(theta)) / I]
    return dydt


def cartpole(y, t, u):
    """
    System of first order equations for a cart-pole system
    The policy commands the force applied to the cart
    (stable equilibrium point with the pole down at [~,0,0,0])
    """

    x, x_dot, theta, theta_dot = y

    m1 = 0.5  # mass of the cart
    m2 = 0.5  # mass of the pendulum
    l = 0.5  # length of the pendulum
    b = 0.1  # friction coefficient
    g = 9.81  # acceleration of gravity

    den = 4 * (m1 + m2) - 3 * m2 * np.cos(theta) ** 2

    dydt = [
        x_dot,
        (
            2 * m2 * l * theta_dot**2 * np.sin(theta)
            + 3 * m2 * g * np.sin(theta) * np.cos(theta)
            + 4 * u
            - 4 * b * x_dot
        )
        / den,
        theta_dot,
        (
            -3 * m2 * l * theta_dot**2 * np.sin(theta) * np.cos(theta)
            - 6 * (m1 + m2) * g * np.sin(theta)
            - 6 * (u - b * x_dot) * np.cos(theta)
        )
        / (l * den),
    ]
    return dydt

def double_pendulum(y, t, u):
    """
    State y: [theta1, theta2, theta_dot1, theta_dot2]
    Input u: [tau1, tau2]
    Time t

    Returns
    Velocities and accelerations dy/dt: [theta_dot1, theta_dot2, alpha1, alpha2]
    """
    pos = y[:2]
    vel = y[2:]
    u = np.append(u, 0)

    I = [0.05, 0.02]
    Ir = 7.659297952841183e-05
    b = [0.001, 0.001]
    coulomb_fric = [0.093, 0.05]
    g = 9.81
    gr = 6.0
    l = [0.4, 0.1]
    m = [0.6, 0.2]
    com = [0.4, 0.1]
    B = np.diag([1, 0])
    #Mass matrix
    m00 = I[0] + I[1] + m[1]*l[0]**2.0 + 2*m[1]*l[0]*com[1]*np.cos(pos[1]) + gr**2.0*Ir + Ir
    m01 = I[1] + m[1]*l[0]*com[1]*np.cos(pos[1]) - gr*Ir
    m10 = I[1] + m[1]*l[0]*com[1]*np.cos(pos[1]) - gr*Ir
    m11 = I[1] + gr**2.0*Ir
    M = np.array([[m00, m01], [m10, m11]])

    C00 = -2*m[1]*l[0]*com[1]*np.sin(pos[1])*vel[1]
    C01 = -m[1]*l[0]*com[1]*np.sin(pos[1])*vel[1]
    C10 = m[1]*l[0]*com[1]*np.sin(pos[1])*vel[0]
    C11 = 0
    C = np.array([[C00, C01], [C10, C11]])
    #Gravity vector
    G0 = -m[0]*g*com[0]*np.sin(pos[0]) - m[1]*g*(l[0]*np.sin(pos[0]) + com[1]*np.sin(pos[0]+pos[1]))
    G1 = -m[1]*g*com[1]*np.sin(pos[0]+pos[1])
    G = np.array([G0, G1])
    #Coulomb vector
    F = np.zeros(2)
    for i in range(2):
        F[i] = b[i]*vel[i] + coulomb_fric[i]*np.arctan(100*vel[i])

    Minv = np.linalg.inv(M)

    force = G + B.dot(u) - C.dot(vel)

    friction = F

    accn = Minv.dot(force - friction)

    alpha1 = accn[0]
    alpha2 = accn[1]

    dydt = [vel[0], vel[1], alpha1, alpha2]

    return dydt
