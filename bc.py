class Boundary: 
    MIN = 0
    MAX = -1
    
    def __init__(self, dim, end, v=None, delta=None):
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

class NoSlipBoundary(Boundary):
    def apply(self, arr):
        if self.end == Boundary.MAX:
            idx = get_slice_indexer(arr, self.dim, -1)
            idx_n = get_slice_indexer(arr, self.dim, -2)       
        elif self.end == Boundary.MIN:
            idx = get_slice_indexer(arr, self.dim, 0)
            idx_n = get_slice_indexer(arr, self.dim, 1)

        arr[idx] = arr[idx_n]
