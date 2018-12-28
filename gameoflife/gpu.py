from gameoflife.main import GameOfLife
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
from string import Template
import sys, os

class GPUOfLife(GameOfLife):

    def __init__(self, matrix=[], shape=(1,1)):
        kernel = read_kernel(width=shape[0], height=shape[1])
        self.fn_step = SourceModule(kernel).get_function('step')

        self.n_block = 16
        self.n_grid = shape[0] // self.n_block + 1
        self.n = self.n_grid * self.n_block

        super().__init__(matrix, shape)

    def step(self):
        matrix = self.pad_matrix()
        out = np.zeros_like(matrix)
        n, m = self.matrix.shape
        block = (self.n_block, self.n_block, 1)
        grid = (self.n_grid, self.n_grid)
        self.fn_step(drv.In(matrix),
                     drv.Out(out),
                     block=block,
                     grid=grid)
        self.matrix = out[1:-1,1:-1] # de-pad

def read_kernel(width, height):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, '../kernels/gameoflife.cu')
    s =  open(filename, 'r').read()
    d = { 'MATRIX_WIDTH': width, 'MATRIX_HEIGHT': height }
    return Template(s).substitute(d)
