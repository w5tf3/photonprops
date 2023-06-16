import numpy as np
from qutip.parallel import parallel_map

def concurrence(rho):
    T_matrix = np.flip(np.diag([-1.,1.,1.,-1.]),axis=1)  # antidiagonal matrix
    M_matrix = np.dot(rho,np.dot(T_matrix,np.dot(np.conjugate(rho),T_matrix)))
    _eigvals = np.real(np.linalg.eigvals(M_matrix))
    _eigvals = np.sqrt(np.sort(_eigvals))
    return np.max([0.0,_eigvals[-1]-np.sum(_eigvals[:-1])])

def calc_concurrence(system,options,return_photoncounts=False, return_rho=False):
    options = {"options": options}
    args = ["ax.dag();ax.dag()*ax;ax",   
            "ay.dag();ay.dag()*ay;ay",
            "ax.dag();ay.dag()*ay;ax",
            "ay.dag();ax.dag()*ax;ay",

            "ax.dag();ax.dag()*ay;ax",
            "ax.dag();ax.dag()*ax;ay",
            "ax.dag();ax.dag()*ay;ay",
        
            "ax.dag();ay.dag()*ax;ay",
            "ax.dag();ay.dag()*ay;ay",
        
            "ay.dag();ax.dag()*ay;ay"]

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
    if return_rho:
        return concurrence(density_matrix/norm), density_matrix
    if return_photoncounts:
        return concurrence(density_matrix/norm), np.real(g2s[0]), np.real(g2s[1]), np.real(g2s[2]), np.real(g2s[3])
    return concurrence(density_matrix/norm)

def calc_concurrence_reuse(system,options,return_photoncounts=False, return_rho=False):
    options = {"options": options}
    # only need to calculate 3 of the 10 g2s by reusing the operators applied at "t"
    # i.e, for all combinations where a_op and c_op are the same.
    args = ["ax.dag();[ax.dag()*ax,ay.dag()*ay,ax.dag()*ay];ax",
            "ay.dag();[ay.dag()*ay,ax.dag()*ax,ax.dag()*ay];ay",
            "ax.dag();[ax.dag()*ax,ax.dag()*ay,ay.dag()*ax,ay.dag()*ay];ay"]
    g2s = parallel_map(system,args,task_kwargs=options)
    density_matrix = np.zeros([4,4], dtype=complex)
    density_matrix[0,0] = np.real(g2s[0][0])  # xx,xx
    density_matrix[3,3] = np.real(g2s[1][0])  # yy,yy
    density_matrix[1,1] = np.real(g2s[0][1])  # xy,xy
    density_matrix[2,2] = np.real(g2s[1][1])  # yx,yx

    density_matrix[0,1] = g2s[0][2]  # xx,xy
    density_matrix[1,0] = np.conj(density_matrix[0,1])
    density_matrix[0,2] = g2s[2][0]  # xx,yx
    density_matrix[2,0] = np.conj(density_matrix[0,2])
    density_matrix[0,3] = g2s[2][1]  # xx,yy
    density_matrix[3,0] = np.conj(density_matrix[0,3])
    
    density_matrix[1,2] = g2s[2][2]  # xy,yx
    density_matrix[2,1] = np.conj(density_matrix[1,2])
    density_matrix[1,3] = g2s[2][3]  # xy,yy
    density_matrix[3,1] = np.conj(density_matrix[1,3])
    
    density_matrix[2,3] = g2s[1][2]  # yx,yy
    density_matrix[3,2] = np.conj(density_matrix[2,3])

    norm = np.trace(density_matrix)
    if return_rho:
        return concurrence(density_matrix/norm), density_matrix
    if return_photoncounts:
        return concurrence(density_matrix/norm), np.real(g2s[0]), np.real(g2s[1]), np.real(g2s[2]), np.real(g2s[3])
    return concurrence(density_matrix/norm)
