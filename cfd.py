import numpy as np
from collections import defaultdict

def pressure_poisson_2d_step(p, u, v, rho, dt, dx, dy, pout, b=None):
    if b is None:
        b = np.zeros_like(p)

    b[1:-1,1:-1] = (rho * (1 / dt * 
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / 
                     (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                      2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                           (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-
                          ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))

    pout[1:-1,1:-1] = (((p[1:-1, 2:] + p[1:-1, 0:-2]) * dy**2 + 
                (p[2:, 1:-1] + p[0:-2, 1:-1]) * dx**2) /
                (2 * (dx**2 + dy**2)) -
                dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1,1:-1])

def momentum_2d_step(u, v, p, rho, nu, f, dt, dx, dy, uout, vout):
    uout[1:-1, 1:-1] = (u[1:-1, 1:-1] -
                         u[1:-1, 1:-1] * dt / dx *
                        (u[1:-1, 1:-1] - u[1:-1, 0:-2]) -
                         v[1:-1, 1:-1] * dt / dy *
                        (u[1:-1, 1:-1] - u[0:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         nu * (dt / dx**2 *
                        (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, 0:-2]) +
                         dt / dy**2 *
                        (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[0:-2, 1:-1])) +
                        f[0] * dt)

    vout[1:-1,1:-1] = (v[1:-1, 1:-1] -
                    u[1:-1, 1:-1] * dt / dx *
                    (v[1:-1, 1:-1] - v[1:-1, 0:-2]) -
                    v[1:-1, 1:-1] * dt / dy *
                    (v[1:-1, 1:-1] - v[0:-2, 1:-1]) -
                    dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                    nu * (dt / dx**2 *
                    (v[1:-1, 2:] - 2 * v[1:-1, 1:-1] + v[1:-1, 0:-2]) +
                    dt / dy**2 *
                    (v[2:, 1:-1] - 2 * v[1:-1, 1:-1] + v[0:-2, 1:-1])) +
                    f[1] * dt)

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

        self._x = [ np.zeros_like(self.p) for _ in range(2) ]

        self.bcs = defaultdict(list)

    def add_boundary_conditions(self, name, bcs):
        self.bcs[name] += bcs

    def get_boundary_conditions(self, name):
        return self.bcs[name]

    def solve(self, dt, its, p_tol, p_max_its, cb=None):
        
        for _ in range(its):
            self.solve_pressure_poisson(dt, tol=p_tol, max_its=p_max_its)
            self.solve_momentum(dt)

            if cb:
                cb()

    def solve_pressure_poisson(self, dt, tol, max_its):
        p, pn, b = self.p, self._x[0], self._x[1]

        dx,dy = self.space.delta
        u,v = self.u, self.v
        bcs = self.get_boundary_conditions('p')

        res = float("inf")

        it = 0
        while res > tol and it < max_its:
            pressure_poisson_2d_step(p, u, v, self.rho, dt, dx, dy, pout=pn, b=b)

            for bc in bcs:
                bc.set_boundary(pn)

            res = np.linalg.norm(p-pn)

            np.copyto(p, pn)
            it += 1

    def solve_momentum(self, dt):
        dx, dy = self.space.delta
        u, un = self.u, self._x[0]
        v, vn = self.v, self._x[1]

        momentum_2d_step(u, v, self.p, self.rho, self.nu, self.f, dt, dx, dy, uout=un, vout=vn)
        
        for bc in self.get_boundary_conditions('u'):
            bc.set_boundary(un)

        for bc in self.get_boundary_conditions('v'):
            bc.set_boundary(vn)

        np.copyto(u, un)
        np.copyto(v, vn)

class Boundary: 
    MIN = 0
    MAX = -1
    
    def __init__(self, v, dim, end, delta=None):
        self.v = v
        self.dim = dim
        self.end = end
        self.delta = delta 
    
    def set_boundary(self, arr): pass

def get_slice_indexer(arr, dim, ind):
    ii = [ slice(None) ] * arr.ndim
    ii[dim] = ind
    return tuple(ii)

class DirichletBoundary(Boundary):
    def set_boundary(self, arr):
        idx = get_slice_indexer(arr, self.dim, self.end)
        arr[idx] = self.v

class NeumannBoundary(Boundary):
    def set_boundary(self, arr):
        if self.end == Boundary.MAX:
            idx = get_slice_indexer(arr, self.dim, -1)
            idx_n = get_slice_indexer(arr, self.dim, -2)       
        elif self.end == Boundary.MIN:
            idx = get_slice_indexer(arr, self.dim, 0)
            idx_n = get_slice_indexer(arr, self.dim, 1)

        arr[idx] = arr[idx_n] - self.v * self.delta

import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot(fluid, fig=None):
    if fig:
        plt.figure(fig.number)
        plt.clf()

    X,Y = fluid.space.coords
    u,v,p = fluid.u, fluid.v, fluid.p

    plt.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)  
    plt.colorbar()

    # plotting the pressure field outlines
    # plt.contour(X, Y, p, cmap=cm.viridis)  

    # plotting velocity field
    plt.streamplot(X, Y, u, v) 
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim([0,2])
    plt.ylim([0,2])

def cavity_flow():
    space = Space([81,81], [2,2])
    fluid = Fluid(space, rho=1, nu=.1)

    fluid.add_boundary_conditions('u', [
        DirichletBoundary(dim=0, v=0, end=Boundary.MIN),
        DirichletBoundary(dim=0, v=1, end=Boundary.MAX), # lid driven 
        DirichletBoundary(dim=1, v=0, end=Boundary.MIN),
        DirichletBoundary(dim=1, v=0, end=Boundary.MAX),
    ])

    fluid.add_boundary_conditions('v', [
        DirichletBoundary(dim=0, v=0, end=Boundary.MIN),
        DirichletBoundary(dim=0, v=0, end=Boundary.MAX),
        DirichletBoundary(dim=1, v=0, end=Boundary.MIN),
        DirichletBoundary(dim=1, v=0, end=Boundary.MAX)
    ])

    fluid.add_boundary_conditions('p', [
        NeumannBoundary(dim=1, v=0, end=Boundary.MAX, delta=space.delta[1]),
        NeumannBoundary(dim=0, v=0, end=Boundary.MIN, delta=space.delta[0]),
        NeumannBoundary(dim=1, v=0, end=Boundary.MIN, delta=space.delta[1]),
        DirichletBoundary(dim=0, v=0, end=Boundary.MAX, delta=space.delta[0])
    ])
    
    fluid.solve(dt=0.001, its=700, p_tol=5e-2, p_max_its=100)
    
    fig, ax = plt.subplots()
    plot(fluid, fig)
    plt.show()#, cb=lambda: plot(fluid,fig))

def membrane():
    space = Space([81,81], [2,2])
    fluid = Fluid(space, rho=1, nu=.1)

    fluid.add_boundary_conditions('u', [
        DirichletBoundary(dim=0, v=0, end=Boundary.MIN),
        DirichletBoundary(dim=0, v=1, end=Boundary.MAX), # lid driven 
        DirichletBoundary(dim=1, v=0, end=Boundary.MIN),
        DirichletBoundary(dim=1, v=0, end=Boundary.MAX),
    ])

    fluid.add_boundary_conditions('v', [
        DirichletBoundary(dim=0, v=0, end=Boundary.MIN),
        DirichletBoundary(dim=0, v=0, end=Boundary.MAX),
        DirichletBoundary(dim=1, v=0, end=Boundary.MIN),
        DirichletBoundary(dim=1, v=0, end=Boundary.MAX)
    ])

    fluid.add_boundary_conditions('p', [
        NeumannBoundary(dim=1, v=0, end=Boundary.MAX, delta=space.delta[1]),
        NeumannBoundary(dim=0, v=0, end=Boundary.MIN, delta=space.delta[0]),
        NeumannBoundary(dim=1, v=0, end=Boundary.MIN, delta=space.delta[1]),
        DirichletBoundary(dim=0, v=0, end=Boundary.MAX, delta=space.delta[0])
    ])
    
    fluid.solve(dt=0.001, its=700, p_tol=5e-2, p_max_its=100)
    
    fig, ax = plt.subplots()
    plot(fluid, fig)
    plt.show()#, cb=lambda: plot(fluid,fig))

if __name__ == "__main__": cavity_flow()
    
    
    
