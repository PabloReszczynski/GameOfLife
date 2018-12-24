import numpy as np

class GameOfLife():
    def __init__(self, matrix=[], shape=None):
        self.matrix = np.array(matrix)
        if shape:
            self.matrix = self.matrix.reshape(shape)

    def __getitem__(self, key):
        return self.matrix.__getitem__(key)

    def neighbour_count(self, x, y):
        count = 0
        matrix = np.pad(self.matrix, (1,1), 'constant', constant_values=(0,))
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1: continue
                count += matrix[x + i, y + j]
        return count

    def step_cell(self, x, y):
        neighbours = self.neighbour_count(x, y)
        cell = self.matrix[x,y]
        if cell:
            if neighbours < 2 or neighbours > 3:
                return 0
        else:
            if neighbours == 3:
                return 1
        return cell

    def step(self):
        n, m = self.matrix.shape
        matrix = np.copy(self.matrix)
        for i in range(n):
            for j in range(m):
                matrix[i,j] = self.step_cell(i, j)

        self.matrix = matrix

