import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from qutip import *
import scipy
from scipy.optimize import fsolve
from scipy.special import j0, jv
from scipy.stats import binom, multinomial

from ms_fidelity import ms_fidelity_info_from_data


def asymmetric_single_mode_ion_ham(eta, mu, omega, psi, Omega_p, Omega_m, delta_omega, N_cutoff):
    H_osc = omega*tensor(num(N_cutoff), qeye(2), qeye(2))
    H_sigmaz = delta_omega/2*(tensor(qeye(N_cutoff), sigmaz(), qeye(2)) + tensor(qeye(N_cutoff), qeye(2), sigmaz()))

    q = create(N_cutoff) + destroy(N_cutoff)
    exp_ietaq       = Qobj(scipy.linalg.expm(1j*eta*np.array(q)))
    
    return QobjEvo([H_osc + H_sigmaz,
                   [tensor(exp_ietaq, sigmap(), qeye(2))  
                  + tensor(exp_ietaq, qeye(2), sigmap()), '1/2*Omega_p*exp(-1j*mu*t - 1j*psi) + 1/2*Omega_m*exp(1j*mu*t + 1j*psi)'],
                   [tensor(exp_ietaq.dag(), sigmam(), qeye(2))  
                  + tensor(exp_ietaq.dag(), qeye(2), sigmam()), '1/2*Omega_p*exp(1j*mu*t + 1j*psi) + 1/2*Omega_m*exp(-1j*mu*t - 1j*psi)']],
            args={'Omega_p' : Omega_p, 'Omega_m' : Omega_m, 'mu' : mu, 'psi' : psi})


def get_probabilities_from_ms_simulation(t_range, *ms_args, **ms_kwargs):
    ham = asymmetric_single_mode_ion_ham(*ms_args, **ms_kwargs)
    N_cutoff = ham.to_list()[0].dims[0][0]
    psi_0 = tensor(fock(N_cutoff, 0), fock(2,0), fock(2,0))
    result = sesolve(ham, psi_0, t_range)
    rhos = [s.ptrace([1,2]) for s in result.states]
    
    p_00 = np.array([abs(r[0,0]) for r in rhos]) 
    p_01 = np.array([abs(r[2,2]) for r in rhos]) 
    p_10 = np.array([abs(r[1,1]) for r in rhos]) 
    p_11 = np.array([abs(r[3,3]) for r in rhos])

    return p_00, p_01, p_10, p_11


def get_sampled_ms_results(t_range, n_loops, *ms_args, **ms_kwargs):
    p_00, p_01, p_10, p_11 = get_probabilities_from_ms_simulation(t_range, *ms_args, **ms_kwargs)

    probs = np.array([p_00, p_01, p_10, p_11])
    numbers = np.array([multinomial.rvs(n_loops, p) for p in probs.T]).T
    N_00, N_01, N_10, N_11 = numbers

    return N_00, N_01, N_10, N_11


def rotated_sigmaz(theta, phi):
    return np.cos(theta)*sigmaz() - np.cos(phi)*np.sin(theta)*sigmax() - np.sin(phi)*np.sin(theta)*sigmay()


def rotated_parity(theta, phi):
    rsigmaz = rotated_sigmaz(theta, phi)
    return tensor(rsigmaz, rsigmaz)


def parity_arr_from_rho(rho, phi_arr, theta):
    parity_arr = np.array([expect(rotated_parity(theta, phi), rho) for phi in phi_arr])
    return parity_arr


def get_parity_probabilities(t, phi_range, theta, *ms_args, **ms_kwargs):
    ham = asymmetric_single_mode_ion_ham(*ms_args, **ms_kwargs)
    N_cutoff = ham.to_list()[0].dims[0][0]
    psi_0 = tensor(fock(N_cutoff, 0), fock(2,0), fock(2,0))
    t_range = np.linspace(0, t, 1001)
    result = sesolve(ham, psi_0, t_range)
    rho_fin = result.states[-1].ptrace([1,2])
    probs = np.diag(rho_fin)

    return probs, parity_arr_from_rho(rho_fin, phi_range, theta)


def get_sampled_parity_results(t, n_loops_diag, n_loops_parity, phi_range, theta, *ms_args, **ms_kwargs):
    probs, parity_arr = get_parity_probabilities(t, phi_range, theta, *ms_args, **ms_kwargs)

    N_00, N_01, N_10, N_11 = multinomial.rvs(n_loops_diag, probs)

    parity_numbers = np.array([binom.rvs(n_loops_parity, p) for p in (parity_arr + 1)/2])

    return np.array([N_00, N_01, N_10, N_11]), parity_numbers


def ms_probabilities(t_i, t_f, dt, n_loops, *ms_args, **ms_kwargs):
    t_range = np.arange(t_i, t_f, dt)
    N_00, N_01, N_10, N_11 = get_sampled_ms_results(t_range, n_loops, *ms_args, **ms_kwargs)

    res_df = pd.DataFrame({'t' : t_range, 'p_00' : N_00, 'p_01' : N_01, 'p_10' : N_10, 'p_11' : N_11})

    plt.plot(t_range, N_00/n_loops, label='$P_{00}$')
    plt.plot(t_range, (N_01+N_10)/n_loops, label='$P_{01} + P_{10}$')
#    plt.plot(t_range, N_10/n_loops, label='$P_{10}$')
    plt.plot(t_range, N_11/n_loops, label='$P_{11}$')

    plt.xlabel('time')

    plt.legend()
    plt.margins(0)
    plt.legend()
    plt.ylim(0, 1)
    plt.show()

    return res_df


def ms_parity_results(t, phi_i, phi_f, delta_phi, n_loops_diag, n_loops_parity, theta, *ms_args, **ms_kwargs):
    phi_range = np.arange(phi_i, phi_f, delta_phi)
    numbers, parity_numbers = get_sampled_parity_results(t, n_loops_diag, n_loops_parity, phi_range, theta, *ms_args, **ms_kwargs)

    res_df = pd.DataFrame({'phi' : phi_range, 'parity' : parity_numbers})

    N_00, N_01, N_10, N_11 = numbers
    print(f"""N_00 = {N_00}\nN_01 = {N_01}\nN_01 = {N_10}\nN_11 = {N_11}\n""")

    plt.plot(phi_range/2/np.pi, parity_numbers/n_loops_parity, label='parity')
    plt.xlabel('$\phi/(2\pi)$')
    plt.legend()
    plt.ylim(0, 1)
    plt.margins(0)
    plt.show()

    return numbers, res_df


class MSSimulator:
    @classmethod
    def random_init(cls, N_cutoff):
        nu = 1.2 + 0.05*np.random.normal()
        omega = 2*np.pi*nu
        Ca_qubit_wl = 729.147*1e-9
        Ca_mass = 39.962591*scipy.constants.physical_constants['atomic mass unit']
        eta = 2*np.pi/Ca_qubit_wl*np.sqrt(scipy.constants.hbar/(2*Ca_mass*omega*1e6))/math.sqrt(2)
        delta_nu_q = 0.0*np.random.normal()

        return MSSimulator(nu, eta, delta_nu_q, N_cutoff)

    def __init__(self, nu, eta, delta_nu_q, N_cutoff):
        self.nu = nu
        self.eta = eta
        self.delta_nu_q = delta_nu_q
        self.N_cutoff = N_cutoff

    
    def print_params(self):
        print(json.dumps({
                'nu' : self.nu,
                'eta' : self.eta,
                'delta_nu_q' : self.delta_nu_q,
                'N_cutoff' : self.N_cutoff
            }, indent=4))


    def amplitudes_from_voltages(self, Vp, Vm):
        # TODO: insert nonlinear functions 

        Omega_p = 2*np.pi*0.1*(Vp - 0.24*Vp*Vm)
        Omega_m = 2*np.pi*0.108*(-0.24*Vp*Vm + Vm)

        return Vp, Vm


    def ms_probabilities(self, t_i, t_f, dt, n_loops, psi, Vp, Vm, aom_p_freq, aom_m_freq, mid_freq):
        ms_params_dict = {
                't_i' : t_i,
                't_f' : t_f,
                'dt' : dt,
                'n_loops' : n_loops,
                'psi' : psi,
                'Vp' : Vp,
                'Vm' : Vm,
                'aom_p_freq' : aom_p_freq,
                'aom_m_freq' : aom_m_freq,
                'mid_freq' : mid_freq
                }

        mu = np.pi*(aom_p_freq - aom_m_freq)
        delta_omega = 2*np.pi*(mid_freq - self.delta_nu_q)
        Omega_p, Omega_m = self.amplitudes_from_voltages(Vp, Vm)
    
        omega = 2*np.pi*self.nu
        eta = self.eta

        self.prob_result = ms_probabilities(t_i, t_f, dt, n_loops, eta, mu, omega, psi, Omega_p, Omega_m, delta_omega, self.N_cutoff)

    def parity_results(self, t, phi_i, phi_f, dphi, n_loops_prob, n_loops_parity, theta, psi, Vp, Vm, aom_p_freq, aom_m_freq, mid_freq):
        parity_params_dict = {
                    't' : t,
                    'phi_i' : phi_i,
                    'phi_f' : phi_f,
                    'dphi' : dphi,
                    'n_loops_prob' : n_loops_prob,
                    'n_loops_parity' : n_loops_parity,
                    'psi' : psi,
                    'Vp' : Vp,
                    'Vm' : Vm,
                    'aom_p_freq' : aom_p_freq,
                    'aom_m_freq' : aom_m_freq,
                    'aom_mid_freq' : mid_freq
                }


        mu = np.pi*(aom_p_freq - aom_m_freq)
        delta_omega = 2*np.pi*(mid_freq - self.delta_nu_q)
        Omega_p, Omega_m = self.amplitudes_from_voltages(Vp, Vm)
    
        omega = 2*np.pi*self.nu
        eta = self.eta
        N_cutoff = self.N_cutoff

        self.parity_params_dict = parity_params_dict
        self.parity_result = ms_parity_results(t, phi_i, phi_f, dphi, n_loops_prob, n_loops_parity, 
                theta, eta, mu, omega, psi, Omega_p, Omega_m, delta_omega, N_cutoff)

    def fidelity_info(self, **kwargs):
        n_loops_parity = self.parity_params_dict['n_loops_parity']

        numbers, coherence_df = self.parity_result
        N00, N01, N10, N11 = numbers
        N0 = N00
        N1 = N01 + N10
        N2 = N11

        return ms_fidelity_info_from_data(N0, N1, N2, coherence_df['phi']/2, coherence_df['parity'], [n_loops_parity]*len(coherence_df), **kwargs)
