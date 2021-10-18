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

def cylinder_flow_lbm():
    fluid = cfd.LatticeBoltzmann(N=(100,400), extent=[100,400], rho0=100., tau=0.6)

    def debug(i,f):
        if i % 100 == 0:
            print(i)
            fvis.plot_vorticity(f.vorticity)

    fluid.solve(1,4000,cb=debug)
    fvis.plot_vorticity(fluid.vorticity)


if __name__ == "__main__": 
    #cavity_flow()
    #cavity_flow_fvm()
    cylinder_flow_lbm()
    
    
    
