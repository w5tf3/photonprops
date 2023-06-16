import numpy as np
from qutip.parallel import parallel_map

def concurrence(rho):
    T_matrix = np.flip(np.diag([-1.,1.,1.,-1.]),axis=1)  # antidiagonal matrix
    M_matrix = np.dot(rho,np.dot(T_matrix,np.dot(np.conjugate(rho),T_matrix)))
    _eigvals = np.real(np.linalg.eigvals(M_matrix))
    _eigvals = np.sqrt(np.sort(_eigvals))
    return np.max([0.0,_eigvals[-1]-np.sum(_eigvals[:-1])])

def calc_concurrence(system,options,return_photoncounts=False):
    options = {"options": options}
    args = ["ax.dag(),ax.dag()*ax,ax",   
            "ay.dag(),ay.dag()*ay,ay",
            "ax.dag(),ay.dag()*ay,ax",
            "ay.dag(),ax.dag()*ax,ay",

            "ax.dag(),ax.dag()*ay,ax",
            "ax.dag(),ax.dag()*ax,ay",
            "ax.dag(),ax.dag()*ay,ay",
        
            "ax.dag(),ay.dag()*ax,ay",
            "ax.dag(),ay.dag()*ay,ay",
        
            "ay.dag(),ax.dag()*ay,ay"]

    g2s = parallel_map(system,args,task_kwargs=options)
    density_matrix = np.zeros([4,4], dtype=complex)
    density_matrix[0,0] = np.real(g2s[0])  # xx,xx
    density_matrix[3,3] = np.real(g2s[1])  # yy,yy
    density_matrix[1,1] = np.real(g2s[2])  # xy,xy
    density_matrix[2,2] = np.real(g2s[3])  # yx,yx
    
    density_matrix[0,1] = g2s[4]  # xx,xy
    density_matrix[1,0] = np.conj(density_matrix[0,1])
    density_matrix[0,2] = g2s[5]  # xx,yx
    density_matrix[2,0] = np.conj(density_matrix[0,2])
    density_matrix[0,3] = g2s[6]  # xx,yy
    density_matrix[3,0] = np.conj(density_matrix[0,3])
    
    density_matrix[1,2] = g2s[7]  # xy,yx
    density_matrix[2,1] = np.conj(density_matrix[1,2])
    density_matrix[1,3] = g2s[8]  # xy,yy
    density_matrix[3,1] = np.conj(density_matrix[1,3])
    
    density_matrix[2,3] = g2s[9]  # yx,yy
    density_matrix[3,2] = np.conj(density_matrix[2,3])
    
    norm = np.trace(density_matrix)
    density_matrix = density_matrix / norm
    if return_photoncounts:
        return concurrence(density_matrix), np.real(g2s[0]), np.real(g2s[1]), np.real(g2s[2]), np.real(g2s[3])
    return concurrence(density_matrix)
