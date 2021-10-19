from enum import Enum
import space as cfdsp

class Boundary(Enum):
    MIN = 0
    MAX = 1

class BoundaryCondition: 
    def __init__(self, dim, b, v=None, space=None, stagger_dir=None, stagger_dim=None):
        self.v = v
        self.dim = dim
        self.space = space
        self.b = b

        if stagger_dir is not None:
            self.i_wall, self.i_in = get_staggered_indexers(dim,len(space.N),b,stagger_dir,stagger_dim)
        else:
            self.i_wall, self.i_in = get_indexers(dim,len(space.N),b)
    
    def apply(self, arr): 
        pass

class Dirichlet(BoundaryCondition):
    def apply(self, arr):
        arr[self.i_wall] = self.v

class DirichletGhost(BoundaryCondition):
    def apply(self, arr):
       arr[self.i_wall] = 2.0*self.v - arr[self.i_in]

class Neumann(BoundaryCondition):
    def apply(self, arr):
        # oddly I think this is also true for ghost dimensions
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

def get_staggered_indexers(dim, ndims, boundary, stagger_dir, stagger_dim):
    if stagger_dir == cfdsp.Stagger.NEGATIVE:
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
    else:
        raise NotImplementedError

    return tuple(i_wall), tuple(i_in)