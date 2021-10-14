import cfd
import fvis
import bc

def cavity_flow():
    fluid = cfd.NavierStokesProjectionMethod(N=[91,91], extent=[1,1], rho=1, nu=0.05)
    dx = fluid.space.delta[0]
    u_lid = 1.0 
    cfl_dt = min(0.25*dx*dx/fluid.nu, 4.0*fluid.nu/u_lid/u_lid)

    fluid.add_boundary_conditions('u', [
        bc.DirichletBoundary(dim=0, v=0, end=bc.Boundary.MIN),
        bc.DirichletBoundary(dim=0, v=u_lid, end=bc.Boundary.MAX), # lid driven 
        bc.DirichletBoundary(dim=1, v=0, end=bc.Boundary.MIN),
        bc.DirichletBoundary(dim=1, v=0, end=bc.Boundary.MAX),
    ])

    fluid.add_boundary_conditions('v', [
        bc.DirichletBoundary(dim=0, v=0, end=bc.Boundary.MIN),
        bc.DirichletBoundary(dim=0, v=0, end=bc.Boundary.MAX),
        bc.DirichletBoundary(dim=1, v=0, end=bc.Boundary.MIN),
        bc.DirichletBoundary(dim=1, v=0, end=bc.Boundary.MAX)
    ])

    fluid.add_boundary_conditions('p', [
        bc.NoSlipBoundary(dim=1, end=bc.Boundary.MAX),
        bc.NoSlipBoundary(dim=0, end=bc.Boundary.MIN),
        bc.NoSlipBoundary(dim=1, end=bc.Boundary.MIN),
        bc.DirichletBoundary(dim=0, v=0, end=bc.Boundary.MAX)
    ])
    
    def debug(i, u, v, p):
        pass

    fluid.solve(dt=cfl_dt, its=700, p_tol=1e-3, p_max_its=50, cb=debug)
    
    fvis.streamplot(fluid)

def cavity_flow_fvm():
    fluid = cfd.NavierStokesFVM(N=[7,7],extent=[1,1],nu=0.01,beta=1.1)
    print(fluid.space.centered_coords)


if __name__ == "__main__": 
    #cavity_flow()
    cavity_flow_fvm()
    #membrane()
    
    
    
