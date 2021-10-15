import numpy as np
from collections import defaultdict

def ddx(f, dx):
    result = np.zeros_like(f)
    result[1:-1,1:-1] = (f[1:-1,2:] - f[1:-1,:-2])/2.0/dx
    return result

def ddy(f, dy):
    result = np.zeros_like(f)
    result[1:-1,1:-1] = (f[2:,1:-1] - f[:-2,1:-1])/2.0/dy
    return result
    
def laplacian(f, dx, dy):
    result = np.zeros_like(f)
    result[1:-1,1:-1] = (f[1:-1,2:] - 2.0*f[1:-1,1:-1] + f[1:-1,:-2])/dx/dx \
                      + (f[2:,1:-1] -2.0*f[1:-1,1:-1] + f[:-2,1:-1])/dy/dy
    return result

def div(u,v,dx,dy):
    return ddx(u,dx) + ddy(v,dy)

def pressure_poisson(p, dx, dy, b, tol, max_its, cb=None):
    cb = cb if cb is not None else lambda x: None

    pn = p.copy()
    it = 0
    err = float("inf")
    
    while it < max_its and err > tol:
        cb(p)

        np.copyto(pn, p)
            
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + 
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                          (2 * (dx**2 + dy**2)) -
                          dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                          b[1:-1,1:-1])


        err = np.linalg.norm(p - pn, 2)
        it += 1
        
    return p, err

class Space: pass

class RegularGrid(Space):
    def __init__(self, N, extent):
        super().__init__()

        self.N = np.array(N)
        self.extent = np.array(extent)
        self.delta = self.extent / (self.N - 1)
        self.coords = [ np.linspace(0,e,n) for e,n in zip(self.extent,self.N)]
        self.grid_coords = np.meshgrid(*self.coords)

class StaggeredGrid(Space):
    def __init__(self, N, extent):
        super().__init__()

        self.N = np.array(N)
        self.extent = np.array(extent)
        self.delta = self.extent / self.N
        self.centered_coords = [ np.linspace(d/2.0,e-d/2.0,n,endpoint=True) for d,e,n in zip(self.delta, self.extent, self.N)]
        self.staggered_coords = [ np.linspace(0,e,n+1,endpoint=True) for d,e,n in zip(self.delta, self.extent, self.N) ]
        self.centered_grid_coords = np.meshgrid(*self.centered_coords)
        # self.staggered_grid_coords = 
        # x-staggered coordinates
        #xu,yu = np.meshgrid(xxs, yy)
        # y-staggered coordinates
        #xv,yv = np.meshgrid(xx, yys)

class Fluid:
    def __init__(self, space):
        self.space = space
        self.bcs = defaultdict(list)

    def add_boundary_conditions(self, name, bcs):
        for bc in bcs:
            bctype = bc.pop('type')
            self.bcs[name].append(bctype(ndims=len(self.space.N),**bc))

    def get_boundary_conditions(self, name):
        return self.bcs[name]

    def get_boundary_conditions_fn(self, name):
        bcs = self.get_boundary_conditions(name)

        def x(arr):
            for bc in bcs:
                bc.apply(arr)

        return x   

    def solve(self, dt, cb=None, **kwargs):
        raise NotImplementedError

class NavierStokesProjectionMethod(Fluid):
    def __init__(self, N, extent, rho, nu, f=None):
        super().__init__(RegularGrid(N, extent))
        
        self.rho = rho # density
        self.nu = nu # viscosity
        self.f = np.zeros(2) if f is None else f

        self.u = np.zeros(self.space.N)
        self.v = np.zeros(self.space.N)
        self.p = np.zeros(self.space.N)

        self._x = [ np.zeros_like(self.p) for _ in range(3) ]


    def solve(self, dt, cb=None, its=100, p_tol=1e-3, p_max_its=50):
        cb = cb if cb is not None else lambda a,b,c,d: None 

        u,v,p,uh,vh = self.u, self.v, self.p, self._x[0], self._x[1]
        dx,dy = self.space.delta 

        apply_p_bcs = self.get_boundary_conditions_fn('p')
        apply_u_bcs = self.get_boundary_conditions_fn('u')
        apply_v_bcs = self.get_boundary_conditions_fn('v')

        for i in range(its):
            
            apply_u_bcs(u)
            apply_v_bcs(v)

            # do the x-momentum RHS
            # u rhs: - d(uu)/dx - d(vu)/dy + ν d2(u)
            uRHS = - ddx(u*u,dx) - ddy(v*u,dy) + self.nu*laplacian(u,dx,dy)
            # v rhs: - d(uv)/dx - d(vv)/dy + ν d2(v)
            vRHS = - ddx(u*v,dx) - ddy(v*v,dy) + self.nu*laplacian(v,dx,dy)
            
            uh = u + dt*uRHS
            vh = v + dt*vRHS
            
            # next compute the pressure RHS: prhs = div(un)/dt + div( [urhs, vrhs])
            prhs = div(uh,vh,dx,dy)/dt
            p,err = pressure_poisson(p, dx, dy, prhs,
                                     tol=p_tol, max_its=p_max_its,
                                     cb=apply_p_bcs)
            
            # finally compute the true velocities
            # u_{n+1} = uh - dt*dpdx
            u = uh - dt*ddx(p,dx)
            v = vh - dt*ddy(p,dy)

            cb(i, u, v, p)

        np.copyto(self.u, u)
        np.copyto(self.v, v)

class NavierStokesFVM(Fluid):
    def __init__(self, N, extent, nu, beta):
        super().__init__(space=StaggeredGrid(N, extent))

        self.nu = nu
        self.beta = beta

        # initialize velocities - we stagger everything in the negative direction. A scalar cell owns its minus face, only.
        # Then, for example, the u velocity field has a ghost cell at x0 - dx and the plus ghost cell at lx
        self.u = np.zeros(self.space.N+2) # include ghost cells

        # # same thing for the y-velocity component
        self.v = np.zeros(self.space.N+2) # include ghost cells

        self.ut = np.zeros_like(self.u)
        self.vt = np.zeros_like(self.v)    

        # initialize the pressure
        self.p = np.zeros(self.space.N+2) # include ghost cells

    def solve(self, dt, cb=None, its=100):
        pass