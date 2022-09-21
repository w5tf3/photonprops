import numpy as np
import matplotlib.pyplot as plt
from photonprops.two_level_system.tls import twolevel_system
# import tqdm

x,gx,t = twolevel_system(tau1=2, area1=1*np.pi, alpha1=0, det1=0, tau2=1,alpha2=0, area2=0*np.pi, det2=0, gamma_e=1/100, delay=0, mode="pop")

plt.plot(t,x,label='x')
plt.plot(t,np.abs(gx),label='gx')
plt.xlabel("time in ps")
plt.ylabel("population")
plt.legend()
plt.show()

brightness, g1, g2 = twolevel_system(tau1=1, area1=1*np.pi, alpha1=0, det1=0, tau2=1,alpha2=0, area2=0*np.pi, det2=0, gamma_e=1/100, delay=0, mode="g2")
print("B: {:.4f}, Indist.: {:.4f}, g2(0): {:.4f}".format(brightness, g1, g2))
