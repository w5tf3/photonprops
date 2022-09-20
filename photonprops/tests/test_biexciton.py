import numpy as np
import matplotlib.pyplot as plt
from photonprops.biexciton.biexciton import biexciton_system
# import tqdm

x,y,b,gx,xb,gb,t = biexciton_system(tau1=2, area1=4.04*np.pi, alpha1=0, det1=-2, pol1_x=1, tau2=1,alpha2=0, area2=1*np.pi, det2=-4, delay=100, mode="pop")

plt.plot(t,x,label='x')
plt.plot(t,y,label='y')
plt.plot(t,b,label='b')
plt.plot(t,np.abs(gx),label='gx')
plt.plot(t,np.abs(gb),label='gb')
plt.plot(t,np.abs(xb),label='xb')
plt.xlabel("time in ps")
plt.ylabel("population")
plt.legend()
plt.show()