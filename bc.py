class Boundary: 
    MIN = 0
    MAX = 1

    CENTER = None    
    NEGATIVE = 0
    POSITIVE = 1
    
    def __init__(self, dim, ndims, end, v=None, delta=None, stagger=None):
        self.v = v
        self.delta = delta
        
        if end == Boundary.MAX:
            i0, i1 = -1, -2            
        elif end == Boundary.MIN:
            i0, i1 = 0, 1
        else:
            raise Exception("unknown boundary position")
        
        self.idx0 = get_slice_indexer(dim,ndims,i0)
        self.idx1 = get_slice_indexer(dim,ndims,i1)
    
    def apply(self, arr): pass

class DirichletBoundary(Boundary):
    def apply(self, arr):
        arr[self.idx0] = self.v

class NeumannBoundary(Boundary):
    def apply(self, arr):
        arr[self.idx0] = arr[self.idx1] - self.v * self.delta

class NoSlipBoundary(Boundary):
    def apply(self, arr):
        arr[self.idx0] = arr[self.idx1]

def get_slice_indexer(dim, ndims, ind):
    ii = [ slice(None) ] * ndims
    ii[dim] = ind
    return tuple(ii)
