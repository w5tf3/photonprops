import numpy as np
import qutip as qt
from photonprops.commons.pulse import ChirpedPulse

HBAR = 0.6582173  # meV*ps

def energies(delta_b=4., delta_0=0.):
    # energy levels of the system
    E_X = -delta_0/2
    E_Y =  delta_0/2
    E_B = -delta_b
    return E_X, E_Y, E_B

def biexciton_system(tau1=1, tau2=1, area1=1*np.pi, area2=0, det1=0, det2=0, alpha1=0, alpha2=0, pol1_x=1, pol2_x=1, delay=1, delta_b=4, delta_0=0.0, gamma_e=1/100, gamma_b=2/100, epsilon=0.01, dt_1=0.1, dt_2=1, options=qt.Options(atol=1e-7), mode="pop", g2_pol="x"):
    """
    in qutip, every energy has to be provided in 1/ps
    Here, a rotating frame with the unsplit exciton energy is chosen. 
    tau1/2: pulse 1/2 duration in ps
    area1/2: pulsearea of pulse 1/2
    det1/2: detuning of pulse 1/2 to unsplit exciton energy in meV
    alpha1/2: chirp of pulse1/2 in ps^2
    pol1/2_x: x polarization component of pulse 1/2. possible options = 0,...,1
    delay: delay of pulse 2 to pulse 1 in ps
    delta_b: biexciton binding in meV
    delta_0: exciton X/Y splitting in meV
    gamma_e: inverse exciton lifetime in 1/ps
    gamma_b: inverse biexciton lifetime in 1/ps
    epsilon: exponential decay, until epsilon is reached
    dt_1: timestep during pulse (0,..,8tau)
    dt_2: timestep after the pulse, during the decay
    mode: "pop" for population or "g2" for g2
    g2_pol: "x" or "y", g2 of x or y polarized light
    """
    delta_b = delta_b / HBAR  # delta_b in 1/ps
    delta_0 = delta_0 / HBAR
    gamma_b = gamma_b / 2  # both X and Y decay. the input gamma_b is 1/tau_b, where tau_b is the lifetime of the biexciton

    # system states
    g = qt.basis(4,0)
    x = qt.basis(4,1)
    y = qt.basis(4,2)
    b = qt.basis(4,3)
    # print(g*g.dag())  # should be 4x4 Matrix with 1.0 at (0,0)

    # number operators
    n_x = x * x.dag()
    n_y = y * y.dag()
    n_b = b * b.dag()

    # transition operators / polarizations
    p_gx = g * x.dag()
    p_gy = g * y.dag()
    p_xb = x * b.dag()
    p_yb = y * b.dag()
    # this one is not needed for the hamiltonian
    p_gb = g * b.dag()

    # collapse operators / spontaneous emission
    c_ops = [np.sqrt(gamma_e) * p_gx, np.sqrt(gamma_e) * p_gy, np.sqrt(gamma_b) * p_xb, np.sqrt(gamma_b) * p_yb]

    # system Hamiltonian
    E_X, E_Y, E_B = energies(delta_b=delta_b, delta_0=delta_0) # note they are already divided by HBAR
    H_sys = E_X * n_x + E_Y * n_y + E_B * n_b

    # pulse 1 and 2, right now assume delay > 0
    tau11=np.sqrt(alpha1**2 / tau1**2 + tau1**2)
    tau22=np.sqrt(alpha2**2 / tau2**2 + tau2**2)
    # choose the longer of the two
    t_start1 = 4*tau11 if tau11 > tau22 else 4*tau22
    # further delay pulse 2
    t_start2 = t_start1 + delay
    pulse1 = ChirpedPulse(tau1, det1, alpha1, t0=t_start1, e0=area1, polar_x=pol1_x)
    pulse2 = ChirpedPulse(tau2, det2, alpha2, t0=t_start2, e0=area2, polar_x=pol2_x)

    # excitation Hamiltonians (daggered, as expressed by polarization operators)
    H_x_dag = -0.5 * (p_gx + p_xb)  # this has to be paired with the conjugated total x-field 
    H_y_dag = -0.5 * (p_gy + p_yb)
    # print(H_x_dag.dag())

    H = [H_sys, [H_x_dag, lambda t,args : np.conj(pulse1.polar_x*pulse1.get_total(t) + pulse2.polar_x*pulse2.get_total(t))],
                [H_y_dag, lambda t,args : np.conj(pulse1.polar_y*pulse1.get_total(t) + pulse2.polar_y*pulse2.get_total(t))],
                [H_x_dag.dag(), lambda t,args : pulse1.polar_x*pulse1.get_total(t) + pulse2.polar_x*pulse2.get_total(t)],
                [H_y_dag.dag(), lambda t,args : pulse1.polar_y*pulse1.get_total(t) + pulse2.polar_y*pulse2.get_total(t)]]
    
    # time axes. has to start at 0 due to limitations in the function calculating the 2-time quantities
    # two different time steps are used: a small dt_1, during the time the pulses are active, and a larger dt_2 during the decay 
    # time axis during the pulses
    t_off = t_start2 + t_start1  # time window where pulse 1 or 2 is still active
    rate = 2*gamma_b if 2*gamma_b<gamma_e else gamma_e
    t_end = t_off - 1/rate *np.log(epsilon)  # note that log(epsilon) is in general negative
    t_axis1 = np.arange(0, t_off, dt_1)
    t_axis2 = np.arange(t_off, t_end, dt_2)  # note that arange does not include the final value so t_off is not in both arrays
    t_axis = np.append(t_axis1, t_axis2)

    if mode == "pop":
        x_occ, y_occ, b_occ, polar_gx, polar_xb, polar_gb = qt.mesolve(H, g, t_axis, c_ops=c_ops, e_ops=[n_x, n_y, n_b, p_gx, p_xb, p_gb], options=options).expect
        return x_occ, y_occ, b_occ, polar_gx, polar_xb, polar_gb, t_axis
    elif mode == "g2":
        tau_axis = t_axis
        sm = p_gx
        _n = n_x
        if g2_pol == "y":
            sm = p_gy
            _n = n_y

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
        # g2=0
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
