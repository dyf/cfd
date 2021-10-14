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

class Space:
    def __init__(self, N, extent):
        self.N = np.array(N)
        self.extent = np.array(extent)
        self.delta = self.extent / (self.N - 1)
        self.coords = np.meshgrid(*[ np.linspace(0,e,n) for e,n in zip(self.extent,self.N)])

class Fluid:
    def __init__(self, space, rho, nu, f=None):
        self.space = space
        self.rho = rho # density
        self.nu = nu
        self.f = np.zeros(2) if f is None else f

        self.u = np.zeros(space.N)
        self.v = np.zeros(space.N)
        self.p = np.zeros(space.N)

        self._x = [ np.zeros_like(self.p) for _ in range(3) ]

        self.bcs = defaultdict(list)

    def add_boundary_conditions(self, name, bcs):
        self.bcs[name] += bcs

    def get_boundary_conditions(self, name):
        return self.bcs[name]

    def get_boundary_conditions_fn(self, name):
        bcs = self.get_boundary_conditions(name)

        def x(arr):
            for bc in bcs:
                bc.apply(arr)

        return x      

    def solve(self, dt, its, p_tol, p_max_its, cb=None):
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

class Boundary: 
    MIN = 0
    MAX = -1
    
    def __init__(self, v, dim, end, delta=None):
        self.v = v
        self.dim = dim
        self.end = end
        self.delta = delta 
    
    def apply(self, arr): pass

def get_slice_indexer(arr, dim, ind):
    ii = [ slice(None) ] * arr.ndim
    ii[dim] = ind
    return tuple(ii)

class DirichletBoundary(Boundary):
    def apply(self, arr):
        idx = get_slice_indexer(arr, self.dim, self.end)
        arr[idx] = self.v

class NeumannBoundary(Boundary):
    def apply(self, arr):
        if self.end == Boundary.MAX:
            idx = get_slice_indexer(arr, self.dim, -1)
            idx_n = get_slice_indexer(arr, self.dim, -2)       
        elif self.end == Boundary.MIN:
            idx = get_slice_indexer(arr, self.dim, 0)
            idx_n = get_slice_indexer(arr, self.dim, 1)

        arr[idx] = arr[idx_n] - self.v * self.delta

