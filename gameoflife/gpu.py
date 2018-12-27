from main import GameOfLife
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
import sys, os

class GPUOfLife(GameOfLife):

    def __init__(self, matrix=[], shape=(1,1)):
        kernel = read_kernel()
        self.fn_step = SourceModule(kernel).get_function('step')

        self.n_block = 16
        self.n_grid = len(matrix) // self.n_block
        self.n = self.n_grid * self.n_block

        super.__init__(matrix, shape)

    def step(self):
        out = np.zeros_like(self.matrix)
        matrix = self.pad_matrix()
        n, m = self.matrix.shape
        block = (self.n_block, self.n_block, 1)
        grid = (self.n_grid, self.n_grid, 1)
        self.fn_step(drv.In(matrix),
                     drv.Out(out),
                     block=block,
                     grid=grid)
        self.matrix = out

def read_kernel():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, '../kernels/gameoflife.cu')
    return open(filename, 'r').readlines()


gol = GPUOfLife([1,2,3,4,5,6,7,8,9], shape=(3,3))
gol.step()
print(gol.matrix)
