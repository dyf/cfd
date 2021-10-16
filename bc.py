from enum import Enum
import space

class Boundary(Enum):
    MIN = 0
    MAX = 1

class BoundaryCondition: 
    def __init__(self, dim, b, v=None, space=None):
        self.v = v
        self.space = space

        if isinstance(space, space.StaggeredGrid):
            self.i_wall, self.i_in = get_staggered_indexers()
        else:
            self.i_wall, self.i_in = get_indexers(dim,len(space.N),b)
    
    def apply(self, arr): 
        pass

class Dirichlet(BoundaryCondition):
    def apply(self, arr):
        arr[self.i_wall] = self.v

class Neumann(BoundaryCondition):
    def apply(self, arr):
        arr[self.i_wall] = arr[self.i_in] - self.v * self.space.delta[self.dim]

class NoSlip(BoundaryCondition):
    def apply(self, arr):
        arr[self.i_wall] = arr[self.i_in]

def get_indexers(dim, ndims, boundary):
    i_wall = [ slice(None) ] * ndims
    i_in = [ slice(None) ] * ndims

    if boundary == Boundary.MIN:
        i_wall[dim] = 0
        i_in[dim] = 1
    elif boundary == boundary.MAX:
        i_wall[dim] = -1
        i_in[dim] = -2
        
    return tuple(i_wall), tuple(i_in)

""" 
# u 
#   - horizontal component of velocity (horizontal/left-right = dim 1)
#   - staggered left (negative dim 1)
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

# v
#  - vertical component of velocity (vertical/up-down  = dim 0)
#  - staggered down (negative dim 0)
# left wall
v[1:,0] = 2.0*Vl - v[1:,1]
# right wall
v[1:,-1] = 2.0*Vr - v[1:,-2]
# bottom wall
v[1,1:-1] = Vb
# top wall
v[-1,1:-1] = Vt 
"""

"""
if bdim = staggered dim 
  if min: 1 on dim, 1:-1 e.e., v = c
  if max: -1 on dim, 1:-1 e.e., v = c
if bdim != staggered dim (ghost): 
  if min: 0 on dim, 1: e.e., v = 2c - inside
  if max: -1 on dim, 1: e.e., v = 2c - inside
"""

def get_staggered_indexers(dim, ndims, boundary, stagger_dir, stagger_dim):
    if stagger_dir == Stagger.NEGATIVE:
        if dim == stagger_dim:
            i_wall = [ slice(1,-1) ] * ndims
            i_in = [ slice(1,-1) ] * ndims
            if boundary == Boundary.MIN:
                i_wall[dim] = 1
                i_in[dim] = 2
            elif boundary == Boundary.MAX:
                i_wall[dim] = -1
                i_in[dim] = -2
        else:
            i_wall = [ slice(1,None) ] * ndims
            i_in = [ slice(1,None) ] * ndims

            if boundary == Boundary.MIN:
                i_wall[dim] = 0
                i_in[dim] = 1
            elif boundary == Boundary.MAX:
                i_wall[dim] = -1
                i_in[dim] = -2

    return tuple(i_wall), tuple(i_in)

if __name__ == "__main__":
    iw,ii = get_staggered_indexers(0,2,Boundary.MIN,stagger_dir=Stagger.NEGATIVE, stagger_dim=0)
    print(iw)