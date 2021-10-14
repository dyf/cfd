import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def streamplot(fluid, fig=None):
    x,y = fluid.space.grid_coords

    vel = np.sqrt(fluid.u**2 + fluid.v**2)
    plt.contourf(x,y,fluid.p,cmap='magma')
    plt.colorbar()
    #plt.quiver(fluid.u,fluid.v,color='k')

    plt.streamplot(x,y,fluid.u, fluid.v)
    plt.xlim(x[0,0], x[-1,-1])
    plt.ylim(y[0,0], y[-1,-1])
    plt.show()
