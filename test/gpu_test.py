import unittest
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.driver as drv
from context import GPUOfLife, read_kernel

class GPUTest(unittest.TestCase):
    def test_neighbours_with_empty_board(self):
        kernel = read_kernel(width=3)
        neighbour_count = SourceModule(kernel).get_function('neighbour_count_test')
        result_buffer = np.array([-1], dtype=np.int32)
        matrix_empty = np.array([0,0,0,0,0,
                                 0,0,0,0,0,
                                 0,0,1,0,0,
                                 0,0,0,0,0,
                                 0,0,0,0,0], dtype=np.int32)
        result_empty = neighbour_count(
            drv.Out(result_buffer),
            drv.In(matrix_empty),
            np.int32(12),
            block=(1, 1, 1)
        )
        self.assertEqual(result_buffer[0], 0)

    def test_neighbours_with_empty_board(self):
        kernel = read_kernel(width=3, height=3)
        neighbour_count = SourceModule(kernel).get_function('neighbour_count_test')
        result_buffer = np.array([-1], dtype=np.int32)
        matrix_full = np.array([0,0,0,0,0,
                                0,1,1,1,0,
                                0,1,2,1,0,
                                0,1,1,1,0,
                                0,0,0,0,0], dtype=np.int32)
        neighbour_count(
            drv.Out(result_buffer),
            drv.In(matrix_full),
            np.int32(12),
            block=(1, 1, 1)
        )
        self.assertEqual(result_buffer[0], 8)

    def test_neighbours(self):
        kernel = read_kernel(width=3, height=3)
        neighbour_count = SourceModule(kernel).get_function('neighbour_count_test')
        result_buffer = np.array([-1], dtype=np.int32)
        matrix = np.array([0,0,0,0,0,
                           0,0,0,0,0,
                           0,1,1,1,0,
                           0,0,0,0,0,
                           0,0,0,0,0], dtype=np.int32)
        result = []
        for i in [6,7,8,11,12,13,16,17,18]:
            neighbour_count(
                drv.Out(result_buffer),
                drv.In(matrix),
                np.int32(i),
                block=(1,1,1)
            )
            result.append(result_buffer[0])
        self.assertEqual(result, [2,3,2,
                                  1,2,1,
                                  2,3,2])


    def test_blinker(self):
        matrix = [0,0,0,
                  1,1,1,
                  0,0,0]
        expected = np.array([0,1,0,
                             0,1,0,
                             0,1,0]).reshape(3,3)
        gol = GPUOfLife(matrix=matrix, shape=(3,3))
        gol.step()
        self.assertTrue(np.array_equal(gol.matrix, expected))
