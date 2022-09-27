import numpy as np
import qutip as qt
from photonprops.commons.pulse import ChirpedPulse

HBAR = 0.6582173  # meV*ps

def twolevel_system(tau1=1, tau2=1, area1=1*np.pi, area2=0, det1=0, det2=0, alpha1=0, alpha2=0, delay=1, gamma_e=1/100, epsilon=0.01, dt_1=0.1, dt_2=1, options=qt.Options(atol=1e-7), mode="pop", tend=None):
    """
    in qutip, every energy has to be provided in 1/ps
    Here, a rotating frame with the unsplit exciton energy is chosen. 
    tau1/2: pulse 1/2 duration in ps
    area1/2: pulsearea of pulse 1/2
    det1/2: detuning of pulse 1/2 to unsplit exciton energy in meV
    alpha1/2: chirp of pulse1/2 in ps^2
    pol1/2_x: x polarization component of pulse 1/2. possible options = 0,...,1
    delay: delay of pulse 2 to pulse 1 in ps
    delta_0: exciton X/Y splitting in meV
    gamma_e: inverse exciton lifetime in 1/ps
    gamma_b: inverse biexciton lifetime in 1/ps
    epsilon: exponential decay, until epsilon is reached
    dt_1: timestep during pulse (0,..,8tau)
    dt_2: timestep after the pulse, during the decay
    mode: "pop" for population or "g2" for g2
    g2_pol: "x" or "y", g2 of x or y polarized light
    """

    # system states
    g = qt.basis(2,0)
    x = qt.basis(2,1)

    # print(g*g.dag())  # should be 2x2 Matrix with 1.0 at (0,0)

    # number operators
    n_x = x * x.dag()

    # transition operators / polarizations
    p_gx = g * x.dag()

    # collapse operators / spontaneous emission
    c_ops = [np.sqrt(gamma_e) * p_gx]

    # system Hamiltonian
    E_X = 0.0  # exciton energy in rotating frame
    H_sys = E_X * n_x

    # pulse 1 and 2, right now assume delay > 0
    tau11=np.sqrt(alpha1**2 / tau1**2 + tau1**2)
    tau22=np.sqrt(alpha2**2 / tau2**2 + tau2**2)
    # choose the longer of the two
    t_start1 = 4*tau11 if tau11 > tau22 else 4*tau22
    # further delay pulse 2
    t_start2 = t_start1 + delay
    pulse1 = ChirpedPulse(tau1, det1, alpha1, t0=t_start1, e0=area1)
    pulse2 = ChirpedPulse(tau2, det2, alpha2, t0=t_start2, e0=area2)

    # excitation Hamiltonians (daggered, as expressed by polarization operators)
    H_x_dag = -0.5 * p_gx  # this has to be paired with the conjugated total x-field 
    # print(H_x_dag.dag())

    H = [H_sys, [H_x_dag, lambda t,args : np.conj(pulse1.get_total(t) + pulse2.get_total(t))],
                [H_x_dag.dag(), lambda t,args : pulse1.get_total(t) + pulse2.get_total(t)]]
    
    # time axes. has to start at 0 due to limitations in the function calculating the 2-time quantities
    # two different time steps are used: a small dt_1, during the time the pulses are active, and a larger dt_2 during the decay 
    # time axis during the pulses
    t_off = t_start2 + t_start1  # time window where pulse 1 or 2 is still active
    rate = gamma_e
    t_end = t_off - 1/rate *np.log(epsilon)  # note that log(epsilon) is in general negative
    t_axis1 = np.arange(0, t_off, dt_1)
    t_axis2 = np.arange(t_off, t_end, dt_2)  # note that arange does not include the final value so t_off is not in both arrays
    t_axis = np.append(t_axis1, t_axis2)
    if tend is not None:
        t_axis = np.arange(0, tend, dt_1)

    if mode == "pop":
        x_occ, polar_gx = qt.mesolve(H, g, t_axis, c_ops=c_ops, e_ops=[n_x, p_gx], options=options).expect
        return x_occ, polar_gx, t_axis
    elif mode == "g2":
        tau_axis = t_axis
        sm = p_gx
        _n = n_x

        # operators for 2-time correlations
        # <A(t)B(t+tau)C(t)>
        a_op = sm.dag()
        b_op = sm.dag() * sm
        c_op = sm

        # Expectationvalue of the occupation to calculate the Brightness
        # which we need for normalization
        n_ex = qt.mesolve(H, g, t_axis, c_ops, e_ops=[_n], options=options).expect[0]
        brightness = gamma_e * np.trapz(n_ex, t_axis)

        # two-time correlation
        G2_t_tau = gamma_e ** 2 * qt.correlation_3op_2t(H, g, t_axis, tau_axis, c_ops, a_op, b_op, c_op, solver='me',
                                                      options=options)
        G2_tau = np.abs(np.trapz(G2_t_tau.transpose(), t_axis))
        g2_tau = G2_tau / brightness**2

        g2 = 2 * np.abs(np.trapz(g2_tau, tau_axis))

        a_op = sm.dag()
        b_op = sm
        # estimate for the Indistinguishability, assuming g2 is negligible. See https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.128.093603, Suppl. Material
        G1_t_tau = qt.correlation_2op_2t(H, g, t_axis, tau_axis, c_ops, a_op, b_op, solver='me',
                                                      options=options)
        G1_tau = np.trapz(np.abs(G1_t_tau.transpose())**2, t_axis)
        g1 = 2 * gamma_e**2 * np.trapz(G1_tau, tau_axis) / brightness**2                            
        return brightness, g1, g2
    else:
        print("unsupported mode. choose pop or g2")
        exit(0)
