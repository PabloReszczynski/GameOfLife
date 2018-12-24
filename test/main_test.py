import unittest
from context import GameOfLife
import numpy as np

class MainTest(unittest.TestCase):

    def test_neighbour_count(self):
        matrix3 = [1,1,1,
                   0,0,0,
                   0,0,0]
        gol = GameOfLife(matrix=matrix3, shape=(3,3))
        self.assertEqual(gol.neighbour_count(1,1), 3)
        self.assertEqual(gol.neighbour_count(0,0), 1)
        self.assertEqual(gol.neighbour_count(1,0), 2)

    def test_birth(self):
        matrix3 = [0,1,0,
                   0,0,1,
                   0,1,0]
        gol = GameOfLife(matrix=matrix3, shape=(3,3))
        gol.step()
        self.assertEqual(gol.matrix[1,1], 1)

    def test_blinker(self):
        matrix = [0,0,0,
                  1,1,1,
                  0,0,0]
        expected = np.array([0,1,0,
                             0,1,0,
                             0,1,0]).reshape(3,3)
        gol = GameOfLife(matrix=matrix, shape=(3,3))
        gol.step()
        self.assertTrue(np.array_equal(gol.matrix, expected))


if __name__ == '__main__':
    unittest.main()
