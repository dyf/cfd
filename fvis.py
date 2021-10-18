import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def streamplot(fluid, staggered=False, fig=None):
    x,y = fluid.space.grid_coords

    p = fluid.p[1:-1,1:-1] if staggered else fluid.p
    u = fluid.u[1:-1,1:-1] if staggered else fluid.u
    v = fluid.v[1:-1,1:-1] if staggered else fluid.v

    plt.contourf(x,y,p,cmap='magma')
    plt.colorbar()
    #plt.quiver(fluid.u,fluid.v,color='k')

    plt.streamplot(x,y,u,v)
    plt.xlim(x[0,0], x[-1,-1])
    plt.ylim(y[0,0], y[-1,-1])
    plt.show()

def plot_vorticity(vorticity):
    cmap = plt.cm.bwr.copy()
    cmap.set_bad('black')
    plt.imshow(vorticity, cmap=cmap)
    plt.clim(-.1, .1)
    ax = plt.gca()
    ax.invert_yaxis()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)	
    ax.set_aspect('equal')	
    plt.show()