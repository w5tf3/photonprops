import numpy as np
import qutip as qt
from photonprops.commons.pulse import ChirpedPulse
from concurrent.futures import ThreadPoolExecutor, wait

HBAR = 0.6582173  # meV*ps

def energies(delta_b=4., delta_0=0.):
    # energy levels of the system
    E_X = -delta_0/2
    E_Y =  delta_0/2
    E_B = -delta_b
    return E_X, E_Y, E_B

def g2_b_cav(ops,options):
    a_op,b_op,c_op = ops.split(",")
    options["a_op"] = a_op
    options["b_op"] = b_op
    options["c_op"] = c_op
    options["mode"] = "g2"
    g2,_,_ = biexciton_system_cav(**options)
    return g2

def biexciton_system_cav(tau1=1, tau2=1, area1=1*np.pi, area2=0, det1=0, det2=0, g_cav=0.25, kappa=1.0, alpha1=0, alpha2=0, pol1_x=1, pol2_x=1, t01=10, t02=10, delta_b=4, delta_0=0.0, gamma_e=1/100, gamma_b=2/100, epsilon=0.01, dt_1=0.1, dt_2=1, tend=100, n_phot=1, options=qt.Options(atol=1e-7), mode="pop", exp_t=False,a_op="ax.dag()",b_op="ax.dag()*ax",c_op="ax", abs=False, initial_state="g"):
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
    g = qt.tensor(qt.basis(4,0),qt.basis(n_phot+1,0),qt.basis(n_phot+1,0))
    x = qt.tensor(qt.basis(4,1),qt.basis(n_phot+1,0),qt.basis(n_phot+1,0))
    y = qt.tensor(qt.basis(4,2),qt.basis(n_phot+1,0),qt.basis(n_phot+1,0))
    b = qt.tensor(qt.basis(4,3),qt.basis(n_phot+1,0),qt.basis(n_phot+1,0))
    # print(g*g.dag())  # should be 4x4 Matrix with 1.0 at (0,0)
    ax = qt.tensor(qt.qeye(4),qt.destroy(n_phot+1),qt.qeye(n_phot+1))
    ay = qt.tensor(qt.qeye(4),qt.qeye(n_phot+1),qt.destroy(n_phot+1))

    n_cavx = ax.dag() * ax
    n_cavy = ay.dag() * ay
    # operators for number/polarization ops. containing all photon states
    _g = qt.tensor(qt.basis(4,0),qt.qeye(n_phot+1),qt.qeye(n_phot+1))
    _x = qt.tensor(qt.basis(4,1),qt.qeye(n_phot+1),qt.qeye(n_phot+1))
    _y = qt.tensor(qt.basis(4,2),qt.qeye(n_phot+1),qt.qeye(n_phot+1))
    _b = qt.tensor(qt.basis(4,3),qt.qeye(n_phot+1),qt.qeye(n_phot+1))
    # number operators
    n_g = _g * _g.dag()
    n_x = _x * _x.dag()
    n_y = _y * _y.dag()
    n_b = _b * _b.dag()

    # transition operators / polarizations
    p_gx = _g * _x.dag()
    p_gy = _g * _y.dag()
    p_xb = _x * _b.dag()
    p_yb = _y * _b.dag()
    # this one is not needed for the hamiltonian
    p_gb = _g * _b.dag()

    initial = eval(initial_state)

    # collapse operators / spontaneous emission
    # take care if (ax + ay) or separate
    c_ops = [np.sqrt(kappa) * ax, np.sqrt(kappa) * ay,np.sqrt(gamma_e) * p_gx, np.sqrt(gamma_e) * p_gy, np.sqrt(gamma_e) * p_xb, np.sqrt(gamma_e) * p_yb]
    # c_ops = [np.sqrt(gamma_e) * p_gx, np.sqrt(gamma_e) * p_gy, np.sqrt(gamma_b) * p_xb, np.sqrt(gamma_b) * p_yb]

    # system Hamiltonian
    E_X, E_Y, E_B = energies(delta_b=delta_b, delta_0=delta_0) # note they are already divided by HBAR
    H_sys = E_X * n_x + E_Y * n_y + E_B * n_b + E_B/2 * (n_cavx + n_cavy)

    # pulse 1 and 2, right now assume delay > 0
    tau11=np.sqrt(alpha1**2 / tau1**2 + tau1**2)
    tau22=np.sqrt(alpha2**2 / tau2**2 + tau2**2)
    # choose the longer of the two
    t_start1 = t01 #4*tau11 if tau11 > tau22 else 4*tau22
    # further delay pulse 2
    t_start2 = t02  # t_start1 + delay
    pulse1 = ChirpedPulse(tau1, det1, alpha1, t0=t_start1, e0=area1, polar_x=pol1_x)
    pulse2 = ChirpedPulse(tau2, det2, alpha2, t0=t_start2, e0=area2, polar_x=pol2_x)

    # excitation Hamiltonians (daggered, as expressed by polarization operators)
    H_x_dag = -0.5 * (p_gx + p_xb)  # this has to be paired with the conjugated total x-field 
    H_y_dag = -0.5 * (p_gy + p_yb)
    # print(H_x_dag.dag())

    H_cav = g_cav * (ax.dag() * ((p_gx + p_xb) ) +  ay.dag() *(p_gy + p_yb))
    #H_cav = g_cav * ax.dag() * p_gx + g_cav * ax.dag() * p_xb + g_cav * ay.dag() * p_gy + g_cav * ay.dag() * p_yb
    H_cav_final = H_cav + H_cav.dag()

    H = [H_sys, H_cav_final,
                [H_x_dag, lambda t,args : np.conj(pulse1.polar_x*pulse1.get_total(t) + pulse2.polar_x*pulse2.get_total(t))],
                [H_y_dag, lambda t,args : np.conj(pulse1.polar_y*pulse1.get_total(t) + pulse2.polar_y*pulse2.get_total(t))],
                [H_x_dag.dag(), lambda t,args : pulse1.polar_x*pulse1.get_total(t) + pulse2.polar_x*pulse2.get_total(t)],
                [H_y_dag.dag(), lambda t,args : pulse1.polar_y*pulse1.get_total(t) + pulse2.polar_y*pulse2.get_total(t)]]
    
    # time axes. has to start at 0 due to limitations in the function calculating the 2-time quantities
    # two different time steps are used: a small dt_1, during the time the pulses are active, and a larger dt_2 during the decay 
    # time axis during the pulses
    t_off = np.max([2*t_start2, 2*t_start1])  # time window where pulse 1 or 2 is still active
    rate = 2*gamma_b if 2*gamma_b<gamma_e else gamma_e
    t_end = np.max([t_off, tend])  # 1/rate *np.log(epsilon)  # note that log(epsilon) is in general negative
    t_axis1 = np.arange(0, t_off, dt_1)
    t_axis2 = np.arange(t_off, t_end, dt_2)  # note that arange does not include the final value so t_off is not in both arrays
    t_axis = np.append(t_axis1, t_axis2)
    if exp_t:
        t_exp = np.exp(np.arange(np.log(t_off),np.log(t_end),dt_1))
        t_axis = np.append(t_axis1, t_exp)

    if mode == "pop":
        g_occ, x_occ, y_occ, b_occ, polar_gx, polar_xb, polar_gb, nx, ny = qt.mesolve(H, initial, t_axis, c_ops=c_ops, e_ops=[n_g, n_x, n_y, n_b, p_gx, p_xb, p_gb, n_cavx, n_cavy], options=options).expect
        return g_occ, x_occ, y_occ, b_occ, polar_gx, polar_xb, polar_gb, t_axis, nx, ny, pulse1, pulse2
    elif mode == "g2":
        tau_axis = t_axis

        a_op = eval(a_op)
        b_op = eval(b_op)
        c_op = eval(c_op)
        G2_t_tau = qt.correlation_3op_2t(H, initial, t_axis, tau_axis, c_ops, a_op, b_op, c_op, solver='me',
                                                      options=options)
        if abs:
            G2_t_tau = np.abs(G2_t_tau)
        G2_tau = np.trapz(G2_t_tau.transpose(), t_axis)
        # return G2_tau, tau_axis
        _g2 = np.trapz(G2_tau, tau_axis)
        return _g2, G2_t_tau, tau_axis
        
    else:
        print("unsupported mode. choose pop or g2")
        exit(0)
