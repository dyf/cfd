import numpy as np
from enum import Enum

class Space: pass

class RegularGrid(Space):
    def __init__(self, N, extent):
        super().__init__()

        self.N = np.array(N)
        self.extent = np.array(extent)
        self.delta = self.extent / (self.N - 1)
        self.coords = [ np.linspace(0,e,n) for e,n in zip(self.extent,self.N)]
        self.grid_coords = np.meshgrid(*self.coords)

class Stagger(Enum):
    NEGATIVE = 0
    POSITIVE = 1

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