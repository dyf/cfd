import numpy as np
from collections import defaultdict
import space
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

def ddx(f, dx):
    return (f[1:-1,2:] - f[1:-1,:-2])/2.0/dx

def ddy(f, dy):
    return (f[2:,1:-1] - f[:-2,1:-1])/2.0/dy
    
def laplacian(f, dx, dy):
    return (f[1:-1,2:] - 2.0*f[1:-1,1:-1] + f[1:-1,:-2])/dx/dx + \
           (f[2:,1:-1] -2.0*f[1:-1,1:-1] + f[:-2,1:-1])/dy/dy

def div(u,v,dx,dy):
    return ddx(u,dx) + ddy(v,dy)

def momentum(u, v, dx, dy, nu):
    # u rhs: - d(uu)/dx - d(vu)/dy + ν d2(u)
    # v rhs: - d(uv)/dx - d(vv)/dy + ν d2(v)

    return ( 
        - ddx(u*u,dx) - ddy(v*u,dy) + nu*laplacian(u,dx,dy),
        - ddx(u*v,dx) - ddy(v*v,dy) + nu*laplacian(v,dx,dy)
    )

def momentum_staggered(u, v, dx, dy, nu):
    # do x-momentum - u is of size (nx + 2) x (ny + 2) - only need to do the interior points
    # u is horizontonal component of velocity, dimension 1
    # LL = u[1,2] , UR = u[n,n] 

    ue = 0.5*(u[1:-1, 2:-1] + u[1:-1, 3:  ])
    uw = 0.5*(u[1:-1, 1:-2] + u[1:-1, 2:-1])
    un = 0.5*(u[1:-1, 2:-1] + u[2:,   2:-1])
    us = 0.5*(u[:-2,  2:-1] + u[1:-1, 2:-1])
    vn = 0.5*(v[2:,   1:-2] + v[2:,   2:-1])
    vs = 0.5*(v[1:-1, 1:-2] + v[1:-1, 2:-1])
    
    convection = - (ue**2 - uw**2)/dx - (un*vn - us*vs)/dy
    diffusion = nu * laplacian(u,dx,dy)[:,1:]#[1:-1,2:-1]

    mx = convection + diffusion
                
    # do y-momentum - only need to do interior points
    # v is vertical component of velocity, staggered negative on dimension 0
    # v LL = v[2,1], UR = v[n,n] 

    ve = 0.5*(v[2:-1, 1:-1] + v[2:-1, 2:  ])
    vw = 0.5*(v[2:-1, :-2 ] + v[2:-1, 1:-1])
    vn = 0.5*(v[2:-1, 1:-1] + v[3:,   1:-1])
    vs = 0.5*(v[1:-2, 1:-1] + v[2:-1, 1:-1])
    ue = 0.5*(u[1:-2, 2:  ] + u[2:-1, 2:  ])
    uw = 0.5*(u[1:-2, 1:-1] + u[2:-1, 1:-1])
    
    convection = - (ue*ve - uw*vw)/dx - (vn**2 - vs**2)/dy
    diffusion = nu * laplacian(v,dx,dy)[1:,:]#[2:-1,1:-1]

    my = convection + diffusion

    return mx, my

def pressure_poisson(u, v, dx, dy, dt, tol, max_its, b=None, p=None, bcs=None):
    bcs = bcs if bcs else []

    if b is None:
        b = np.zeros_like(u)

    if p is None:
        p = np.zeros_like(u)

    b[1:-1,1:-1] = div(u,v,dx,dy)/dt

    pn = p.copy()
    it = 0
    err = float("inf")
    
    while it < max_its and err > tol:
        for bc in bcs:
            bc.apply(p)

        np.copyto(pn, p)
            
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + 
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                          (2 * (dx**2 + dy**2)) -
                          dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                          b[1:-1,1:-1])


        err = np.linalg.norm(p - pn, 2)
        it += 1
        
    return p, err

def sparse_pressure_matrix(nx, ny, dx, dy, bc_left, bc_right, bc_bottom, bc_top):

    Ap = np.zeros([ny,nx])
    Ae = 1.0/dx/dx*np.ones([ny,nx])
    As = 1.0/dy/dy*np.ones([ny,nx])
    An = 1.0/dy/dy*np.ones([ny,nx])
    Aw = 1.0/dx/dx*np.ones([ny,nx])

    bc_left.apply(Aw)
    bc_right.apply(Ae)
    bc_bottom.apply(As)
    bc_top.apply(An)

    Ap = -(Aw + Ae + An + As)

    n = nx*ny
    d0 = Ap.reshape(n)
    de = Ae.reshape(n)[:-1]
    dw = Aw.reshape(n)[1:]
    ds = As.reshape(n)[nx:]
    dn = An.reshape(n)[:-nx]
    A1 = scipy.sparse.diags([d0, de, dw, dn, ds], [0, 1, -1, nx, -nx], format='csr')

    return A1

def pressure_poisson_sparse(A1, u, v, dx, dy, dt, nx, ny):
    # do pressure - prhs = 1/dt * div(uhat)
    # we will only need to fill the interior points. This size is for convenient indexing
    divut = np.zeros([ny+2,nx+2]) 
    #divut[1:-1,1:-1] = ddx(ut,dx) + ddy(vt,dy)#div(ut,vt,dx,dy)
    divut[1:-1,1:-1] = (u[1:-1,2:] - u[1:-1,1:-1])/dx + (v[2:,1:-1] - v[1:-1,1:-1])/dy
    
    prhs = 1.0/dt * divut
    
    ###### Use the sparse linear solver
    #     pt = scipy.sparse.linalg.spsolve(A1,prhs[1:-1,1:-1].ravel()) #theta=sc.linalg.solve_triangular(A,d)
    pt,info = scipy.sparse.linalg.bicg(A1,prhs[1:-1,1:-1].ravel(),tol=1e-10) #theta=sc.linalg.solve_triangular(A,d)
    return pt.reshape([ny,nx])

class Fluid:
    def __init__(self, space):
        self.space = space
        self.bcs = defaultdict(list)

    def add_boundary_condition(self, name, bc, **kwargs):
        bctype = bc.pop('type')
        self.bcs[name].append(bctype(space=self.space,**bc,**kwargs))

    def add_boundary_conditions(self, name, bcs, **kwargs):
        for bc in bcs:
            self.add_boundary_condition(name,bc,**kwargs)         

    def get_boundary_conditions(self, name):
        return self.bcs[name]

    def solve(self, dt, cb=None, **kwargs):
        raise NotImplementedError

class NavierStokesProjectionMethod(Fluid):
    def __init__(self, N, extent, rho, nu, f=None):
        super().__init__(space=space.RegularGrid(N, extent))
        
        self.rho = rho # density
        self.nu = nu # viscosity
        self.f = np.zeros(2) if f is None else f

        self.u = np.zeros(self.space.N)
        self.v = np.zeros(self.space.N)
        self.p = np.zeros(self.space.N)

        self._x = [ np.zeros_like(self.p) for _ in range(3) ]


    def solve(self, dt, cb=None, its=100, p_tol=1e-3, p_max_its=50):
        cb = cb if cb is not None else lambda a,b,c,d: None 

        u,v,p,uh,vh,b = self.u, self.v, self.p, self._x[0], self._x[1], self._x[2]

        dx,dy = self.space.delta 

        p_bcs = self.get_boundary_conditions('p')
        u_bcs = self.get_boundary_conditions('u')
        v_bcs = self.get_boundary_conditions('v')

        for i in range(its):
            
            for bc in u_bcs:
                bc.apply(u)
            
            for bc in v_bcs:
                bc.apply(v)

            # do the x-momentum RHS            
            uRHS, vRHS = momentum(u,v,dx,dy,self.nu)
            
            uh[1:-1,1:-1] = u[1:-1,1:-1] + dt*uRHS
            vh[1:-1,1:-1] = v[1:-1,1:-1] + dt*vRHS
                        
            p,err = pressure_poisson(uh, vh, dx, dy, dt, b=b, p=p,
                                     tol=p_tol, max_its=p_max_its,
                                     bcs=p_bcs)
            
            # finally compute the true velocities
            # u_{n+1} = uh - dt*dpdx
            u[1:-1,1:-1] = uh[1:-1,1:-1] - dt*ddx(p,dx)
            v[1:-1,1:-1] = vh[1:-1,1:-1] - dt*ddy(p,dy)

            cb(i, u, v, p)

        np.copyto(self.u, u)
        np.copyto(self.v, v)

class NavierStokesFVM(Fluid):
    def __init__(self, N, extent, nu, beta):
        super().__init__(space=space.StaggeredGrid(N, extent))

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
        ny,nx = self.space.N
        dy,dx = self.space.delta

        u,v,ut,vt,p = self.u, self.v, self.ut, self.vt, self.p

        u_bcs = self.get_boundary_conditions('u')
        v_bcs = self.get_boundary_conditions('v')
        p_bcs = [ self.get_boundary_conditions(x)[0] for x in ('p_left','p_right','p_bottom','p_top')]

        def apply_p_bcs(A_left, A_right, A_bottom, A_top):
            pL.apply(A_left)
            pR.apply(A_right)
            pB.apply(A_bottom)
            pT.apply(A_top)

        A1 = sparse_pressure_matrix(nx, ny, dx, dy, *p_bcs)

        nsteps = 1000
        for n in range(0,nsteps):
            for bc in u_bcs:
                bc.apply(u)

            for bc in v_bcs:
                bc.apply(v)
                
            """  # left wall
            u[1:-1,1] = 0.0
            # right wall
            u[1:-1,-1] = 0.0
            # top wall
            u[-1,1:] = 2.0*Ut - u[-2,1:]
            # bottom wall
            u[0,1:] = 2.0*Ub - u[1,1:]

            # left wall
            v[1:,0] = 2.0*Vl - v[1:,1]
            # right wall
            v[1:,-1] = 2.0*Vr - v[1:,-2]
            # bottom wall
            v[1,1:-1] = 0.0
            # top wall
            v[-1,1:-1] = 0.0      """
        
            mx, my = momentum_staggered(u, v, dx, dy, self.nu)

            ut[1:-1,2:-1] = u[1:-1,2:-1] + dt * mx
            vt[2:-1,1:-1] = v[2:-1,1:-1] + dt * my        
            
            p[:,:] = 0
            p[1:-1,1:-1] = pressure_poisson_sparse(A1, ut, vt, dx, dy, dt, nx, ny)

            # time advance
            u[1:-1,2:-1] = ut[1:-1,2:-1] - dt * (p[1:-1,2:-1] - p[1:-1,1:-2])/dx
            v[2:-1,1:-1] = vt[2:-1,1:-1] - dt * (p[2:-1,1:-1] - p[1:-2,1:-1])/dy  
