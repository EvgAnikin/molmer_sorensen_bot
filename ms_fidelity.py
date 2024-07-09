import itertools
import math
import numpy as np
import pandas as pd
import re

import parameter_estimation as pe


def ms_fidelity_info_from_data(N0, N1, N2, phases, parity_sums, loop_numbers, 
        init_params, parity_fit_type='max_likelihood', fit_function='default'):

    N_tot = N0 + N1 + N2
    N_even = N0 + N2
    N_odd = N1
    P_even = N_even/(N_even + N_odd)
    P_odd = N_odd/(N_even + N_odd)
    delta_P_even = math.sqrt(P_even*P_odd/N_tot)
    
    N_loops = loop_numbers[0]
    if np.any(np.array(loop_numbers) != N_loops):
        raise ValueError('Unequal number of loops per point')
        
    def prob_sym(theta, params):
        theta_0, C = params
        return 0.5 - C*np.cos(4*theta - theta_0)
    
    def deriv_prob_sym(theta, params):
        theta_0, C  = params
        return np.array([-C*np.sin(4*theta - theta_0),
                       -np.cos(4*theta - theta_0)])
    
    def prob_w_linear_shift(theta, params):
        theta_0, C, shift = params
        return 0.5 - C*np.cos(4*theta - theta_0) + shift

    def deriv_prob_w_linear_shift(theta, params):
        theta_0, C, shift = params
        return np.array([-C*np.sin(4*theta - theta_0),
                       -np.cos(4*theta - theta_0),
                        1])

    def prob_w_nshift(theta, params):
        theta_0, C, r = params
        return 0.5 + C*np.cos(4*theta - theta_0) + (0.5 - C)*r
    
    def deriv_prob_w_nshift(theta, params):
        theta_0, C, r = params
        return np.array([C*np.sin(4*theta - theta_0),
                           np.cos(4*theta - theta_0) - r,
                           0.5 - C])


    if fit_function == 'default':
        prob = prob_sym
        deriv_prob = deriv_prob_sym
    elif fit_function == 'with_linear_shift':
        prob = prob_w_linear_shift
        deriv_prob = deriv_prob_w_linear_shift
    elif fit_funttion == 'with_normalized_shift':
        prob = prob_w_nshift
        deriv_prob = deriv_prob_w_nshift
    else:
        raise ValueError('Unsupported fit function')

    
    if parity_fit_type == 'max_likelihood':
        pbd = pe.ParametricBinomialDistr(prob, deriv_prob)
        params, cov = pbd.maximum_likelihood_estimate(parity_sums, phases, N_loops, init_params=init_params)
    elif parity_fit_type == 'mls':
        params = pe.min_least_squares_estimate(prob, parity_sums, phases, N_loops)
        cov = None
    elif parity_fit_type == 'wmls':
        params, cov = pe.weighted_min_least_squares_estimate(prob, parity_sums, phases, N_loops, p0=init_params)
    else:
        raise ValueError('Unsupported fit type')
    
    theta_0, C = params[:2]
    F = P_even/2 + abs(C)
  
    try:
        params_err = np.sqrt(np.diag(cov))
        delta_theta_0, delta_C = params_err[:2]
        delta_F = np.sqrt(delta_C**2 + delta_P_even**2/4)
    except (ValueError, TypeError) as e:
        delta_C = None
        delta_F = None
    
    return {
            'P_even': P_even,
            'P_even_error' : delta_P_even,
            'coherence' : C,
            'coherence_error' : delta_C,
            'fidelity' : F,
            'fidelity_error' : delta_F,
            'theta_0' : theta_0,
            'delta_theta_error' : delta_theta_0,
            'params' : params.tolist(),
            'params_err' : params_err.tolist(),
            'N0' : int(N0),
            'N1' : int(N1),
            'N2' : int(N2)
        }
