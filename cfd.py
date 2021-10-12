import numpy as np
from collections import defaultdict

class Space:
    def __init__(self, N, extent):
        self.N = np.array(N)
        self.extent = np.array(extent)
        self.delta = self.extent / (self.N - 1)
        self.coords = np.meshgrid(*[ np.linspace(0,e,n) for e,n in zip(self.extent,self.N)])

class Fluid:
    def __init__(self, space, rho, nu):
        self.space = space
        self.rho = rho # density
        self.nu = nu

        self.u = np.zeros(space.N)
        self.v = np.zeros(space.N)
        self.p = np.zeros(space.N)

        self._x0 = np.zeros_like(self.p)
        self._x1 = np.zeros_like(self.p)

        self.bcs = defaultdict(list)

    def add_boundary_conditions(self, name, bcs):
        self.bcs[name] += bcs

    def get_boundary_conditions(self, name):
        return self.bcs[name]

    def solve(self, dt, its):
        for _ in range(its):
            self.solve_pressure_poisson(dt, its=50)
            self.solve_momentum(dt)

    def solve_pressure_poisson(self, dt, its):
        p, pn = self.p, self._x0

        dx,dy = self.space.delta
        u,v = self.u, self.v
        bcs = self.get_boundary_conditions('p')

        for i in range(its):
            b = (self.rho * (1 / dt * 
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / 
                     (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                      2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                           (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-
                          ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))

            pn[1:-1,1:-1] = (((p[1:-1, 2:] + p[1:-1, 0:-2]) * dy**2 + 
                          (p[2:, 1:-1] + p[0:-2, 1:-1]) * dx**2) /
                          (2 * (dx**2 + dy**2)) -
                          dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b)

            for bc in bcs:
                bc.set_boundary(pn)

            np.copyto(p, pn)

    def solve_momentum(self, dt):
        dx, dy = self.space.delta
        p = self.p
        u, un = self.u, self._x0
        v, vn = self.v, self._x1

        un[1:-1, 1:-1] = (u[1:-1, 1:-1]-
                         u[1:-1, 1:-1] * dt / dx *
                        (u[1:-1, 1:-1] - u[1:-1, 0:-2]) -
                         v[1:-1, 1:-1] * dt / dy *
                        (u[1:-1, 1:-1] - u[0:-2, 1:-1]) -
                         dt / (2 * self.rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         self.nu * (dt / dx**2 *
                        (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, 0:-2]) +
                         dt / dy**2 *
                        (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[0:-2, 1:-1])))

        vn[1:-1,1:-1] = (v[1:-1, 1:-1] -
                        u[1:-1, 1:-1] * dt / dx *
                       (v[1:-1, 1:-1] - v[1:-1, 0:-2]) -
                        v[1:-1, 1:-1] * dt / dy *
                       (v[1:-1, 1:-1] - v[0:-2, 1:-1]) -
                        dt / (2 * self.rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                        self.nu * (dt / dx**2 *
                       (v[1:-1, 2:] - 2 * v[1:-1, 1:-1] + v[1:-1, 0:-2]) +
                        dt / dy**2 *
                       (v[2:, 1:-1] - 2 * v[1:-1, 1:-1] + v[0:-2, 1:-1])))
        
        for bc in self.get_boundary_conditions('u'):
            bc.set_boundary(un)

        for bc in self.get_boundary_conditions('v'):
            bc.set_boundary(vn)

        np.copyto(u, un)
        np.copyto(v, vn)

class Boundary: 
    MIN = 0
    MAX = -1
    
    def __init__(self, v, dim, end):
        self.v = v
        self.dim = dim
        self.end = end
    
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

        arr[idx] = arr[idx_n]

def plot(fluid):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    X,Y = fluid.space.coords
    u,v,p = fluid.u, fluid.v, fluid.p

    plt.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)  
    plt.colorbar()
    # plotting the pressure field outlines
    plt.contour(X, Y, p, cmap=cm.viridis)  
    qs = 4
    # plotting velocity field
    #plt.quiver(X[::qs, ::qs], Y[::qs, ::qs], u[1:-1:qs, 1:-1:qs], v[1:-1:qs, 1:-1:qs]) 
    plt.streamplot(X, Y, u, v) 
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim([0,2])
    plt.ylim([0,2])

    plt.show()

if __name__ == "__main__":
    space = Space([41,41], [2,2])
    fluid = Fluid(space, rho=1, nu=.1)

    fluid.add_boundary_conditions('u', [
        DirichletBoundary(dim=1, v=0, end=Boundary.MAX),
        DirichletBoundary(dim=0, v=0, end=Boundary.MIN),
        DirichletBoundary(dim=1, v=0, end=Boundary.MIN),
        DirichletBoundary(dim=0, v=1, end=Boundary.MAX), # lid driven 
    ])

    fluid.add_boundary_conditions('v', [
        DirichletBoundary(dim=0, v=0, end=Boundary.MIN),
        DirichletBoundary(dim=0, v=0, end=Boundary.MAX),
        DirichletBoundary(dim=1, v=0, end=Boundary.MIN),
        DirichletBoundary(dim=1, v=0, end=Boundary.MAX)
    ])

    fluid.add_boundary_conditions('p', [
        NeumannBoundary(dim=1, v=0, end=Boundary.MAX),
        NeumannBoundary(dim=0, v=0, end=Boundary.MIN),
        NeumannBoundary(dim=1, v=0, end=Boundary.MIN),
        DirichletBoundary(dim=0, v=0, end=Boundary.MAX)
    ])
    
    fluid.solve(dt=0.001, its=700)

    plot(fluid)

    
    
