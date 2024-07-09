import numpy as np
from scipy.stats import binom
import scipy.optimize
from scipy.optimize import curve_fit

class ParametricBinomialDistr:
    def __init__(self, prob, prob_func_deriv=None):
        """
        Initialize a ParametricBinomialDistr object with a parameter-dependent success probability function prob(a, params)
        """
        self.prob_func = prob
        self.prob_func_deriv = prob_func_deriv

    def get_sample(self, arg_range, N, params):
        return [binom.rvs(N, self.prob_func(a, params)) for a in arg_range]

    def log_likelihood(self, sample, arg_range, N, params):
        p = self.prob_func(arg_range, params)
        if np.all(p > 0)  and np.all(p < 1):
            return np.sum(sample*np.log(p) + (N - sample)*np.log(1 - p))
        else:
            return -1e32

    def maximum_likelihood_estimate(self, sample, arg_range, N, init_params, bounds=None):
        def likelihood_func(x):
            return -self.log_likelihood(sample, arg_range, N, x)

        def likelihood_func_deriv(x):
            p = self.prob_func(arg_range, x)
            dpdv = np.array([self.prob_func_deriv(a, x) for a in arg_range])
            return -(dpdv.T @ (sample/p - (N - sample)/(1-p))).T


        result = scipy.optimize.minimize(likelihood_func, init_params, bounds=bounds, jac=likelihood_func_deriv)
        params = result.x
        cov = self.maximum_likelihood_cov(arg_range, N, params)
        return params, cov

    def maximum_likelihood_cov(self, arg_range, N, params):
        prob = self.prob_func(arg_range, params)
        dpdv = np.array([self.prob_func_deriv(a, params) for a in arg_range])
    
        #print(dpdv)
        gessian = -N*dpdv.T @ np.diag(1/(prob*(1-prob))) @ dpdv
        gessian_inv = np.linalg.inv(gessian)
        
        return N*(gessian_inv @ dpdv.T @ np.diag(1/(prob*(1-prob))) @ dpdv @ gessian_inv.T)


def min_least_squares_estimate(prob_func, sample, theta_arr, N):
    def mls_func(x):
        theta_0, C = x
        p = prob_func(theta_arr, [theta_0, C])
        return np.sum((sample/N - p)**2)
    
    result = scipy.optimize.minimize(mls_func, np.array([0,0.4]), bounds=[(-np.pi, np.pi), (-0.9999999, 0.99999999)])
    return result.x


def weighted_min_least_squares_estimate(prob_func, sample, theta_arr, N, p0=None):
    parity = sample/N
    error = np.array([np.max([(parity[i]*(1-parity[i])/N)**0.5,1/(N+2)]) for i in range(len(sample))])
    
    def prob_func_mod(x, *p):
        return prob_func(x, p)

    parameters, cov_matrix = curve_fit(prob_func_mod, theta_arr, parity, p0=p0, sigma=error)
    return parameters, cov_matrix
