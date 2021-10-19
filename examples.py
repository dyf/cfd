import numpy as np
import cfd
import fvis
import space as sp
import bc

def cavity_flow():
    fluid = cfd.NavierStokesProjectionMethod(N=[91,91], extent=[1,1], rho=1, nu=0.05)
    dx = fluid.space.delta[0]
    u_lid = 1.0 
    cfl_dt = min(0.25*dx*dx/fluid.nu, 4.0*fluid.nu/u_lid/u_lid)

    fluid.add_boundary_conditions('u', [
        dict(type=bc.Dirichlet, dim=0, v=0, b=bc.Boundary.MIN),
        dict(type=bc.Dirichlet, dim=0, v=u_lid, b=bc.Boundary.MAX), # lid driven 
        dict(type=bc.Dirichlet, dim=1, v=0, b=bc.Boundary.MIN),
        dict(type=bc.Dirichlet, dim=1, v=0, b=bc.Boundary.MAX),
    ])

    fluid.add_boundary_conditions('v', [
        dict(type=bc.Dirichlet, dim=0, v=0, b=bc.Boundary.MIN),
        dict(type=bc.Dirichlet, dim=0, v=0, b=bc.Boundary.MAX),
        dict(type=bc.Dirichlet, dim=1, v=0, b=bc.Boundary.MIN),
        dict(type=bc.Dirichlet, dim=1, v=0, b=bc.Boundary.MAX)
    ])

    fluid.add_boundary_conditions('p', [
        dict(type=bc.NoSlip, dim=1, b=bc.Boundary.MAX),
        dict(type=bc.NoSlip, dim=0, b=bc.Boundary.MIN),
        dict(type=bc.NoSlip, dim=1, b=bc.Boundary.MIN),
        dict(type=bc.Dirichlet, dim=0, v=0, b=bc.Boundary.MAX)
    ])
    
    def debug(i, u, v, p):
        pass

    fluid.solve(dt=cfl_dt, its=700, p_tol=1e-3, p_max_its=50, cb=debug)
    
    fvis.streamplot(fluid)

def cavity_flow_fvm():
    fluid = cfd.NavierStokesFVM(N=[41,41],extent=[1,1],nu=0.01,beta=1.1)

    fluid.add_boundary_conditions('u', [
        dict(type=bc.DirichletGhost, dim=0, b=bc.Boundary.MIN, v=0),  # bottom
        dict(type=bc.DirichletGhost, dim=0, b=bc.Boundary.MAX, v=10), # top
        dict(type=bc.Dirichlet,      dim=1, b=bc.Boundary.MIN, v=0),  # left
        dict(type=bc.Dirichlet,      dim=1, b=bc.Boundary.MAX, v=0),  # right
    ], stagger_dir=sp.Stagger.NEGATIVE, stagger_dim=1)

    fluid.add_boundary_conditions('v', [
        dict(type=bc.Dirichlet,      dim=0, b=bc.Boundary.MIN, v=0), # bottom
        dict(type=bc.Dirichlet,      dim=0, b=bc.Boundary.MIN, v=0), # top
        dict(type=bc.DirichletGhost, dim=1, b=bc.Boundary.MIN, v=0), # left
        dict(type=bc.DirichletGhost, dim=1, b=bc.Boundary.MIN, v=0), # right
    ], stagger_dir=sp.Stagger.NEGATIVE, stagger_dim=0)

    fluid.add_boundary_conditions('p', [
        dict(type=bc.Dirichlet, dim=0, b=bc.Boundary.MIN, v=0),
        dict(type=bc.Dirichlet, dim=0, b=bc.Boundary.MAX, v=0),
        dict(type=bc.Dirichlet, dim=1, b=bc.Boundary.MIN, v=0),
        dict(type=bc.Dirichlet, dim=1, b=bc.Boundary.MAX, v=0), 
    ])
    
    fluid.solve(.0001)
    fvis.streamplot(fluid, staggered=True)

import matplotlib.pyplot as plt

def cylinder_flow_lbm():
    np.random.seed(42)
    
    fluid = cfd.LatticeBoltzmann(N=(100,400), extent=[99,399], rho0=100., tau=0.6)

    # initial conditions
    Ny,Nx,NL = fluid.space.N[0], fluid.space.N[1], fluid.space.NL
    fluid.F = 1 + 0.01*np.random.randn(Ny,Nx,NL)
    Y,X = fluid.space.grid_coords
    fluid.F[:,:,3] += 2 * (1+0.2*np.cos(2*np.pi*X/Nx*4))
    fluid.object_mask = (X - Nx/4)**2 + (Y - Ny/2)**2 < (Ny/4)**2

    fig = plt.figure()
    def debug(i,f):
        if i % 10 == 0:
            print(i)
            fvis.plot_vorticity(f.vorticity)
            plt.pause(0.001)

    fluid.solve(1,4000,cb=debug)



if __name__ == "__main__": 
    #cavity_flow()
    #cavity_flow_fvm()
    cylinder_flow_lbm()
    
    
    
