import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gameoflife.main import GameOfLife
from gameoflife.gpu import GPUOfLife, read_kernel
