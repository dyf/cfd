from enum import Enum

class Boundary(Enum):
    MIN = 0
    MAX = 1

class Stagger(Enum):
    NEGATIVE = 0
    POSITIVE = 1
        
""" 
# left wall
u[1:-1,1] = Ul 
u[:,0] = Ul 

# right wall
u[1:-1,-1] = Ur 
u[:,-1] = Ur 

# top wall
u[-1,1:] = 2.0*Ut - u[-2,1:] 
u[-1,:] = Ut 

# bottom wall
u[0,1:] = 2.0*Ub - u[1,1:]
u[0,:] = Ub

# left wall
v[1:,0] = 2.0*Vl - v[1:,1]
# right wall
v[1:,-1] = 2.0*Vr - v[1:,-2]
# bottom wall
v[1,1:-1] = Vb
# top wall
v[-1,1:-1] = Vt 
"""

class BoundaryCondition: 
    def __init__(self, dim, ndims, end, v=None, delta=None):
        self.v = v
        self.delta = delta
        
        if end == Boundary.MAX:
            i0, i1 = -1, -2            
        elif end == Boundary.MIN:
            i0, i1 = 0, 1
        
        self.idx0 = get_slice_indexer(dim,ndims,i0)
        self.idx1 = get_slice_indexer(dim,ndims,i1)
    
    def apply(self, arr): pass

class Dirichlet(BoundaryCondition):
    def apply(self, arr):
        arr[self.idx0] = self.v

class Neumann(BoundaryCondition):
    def apply(self, arr):
        arr[self.idx0] = arr[self.idx1] - self.v * self.delta

class NoSlip(BoundaryCondition):
    def apply(self, arr):
        arr[self.idx0] = arr[self.idx1]

def get_slice_indexer(dim, ndims, ind):
    ii = [ slice(None) ] * ndims
    ii[dim] = ind
    return tuple(ii)
