__device__ int neighbour_count(int *board, int x, int y) {
  int count = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      count += board[x + i + y + j];
    }
  }
  count -= board[x + y];
  return count;
}

__device__ int step_cell(int cell, int neighbours) {
  if (cell) {
    if (neighbours < 2 || neighbours > 3) {
      return 0;
    }
  } else if (neighbours == 3) {
    return 1;
  }
  return cell;
}

__global__ void step(int *board_in, int *board_out) {
  int x = threadIdx.x + blockDim.x * gridDim.x;
  int y = threadIdx.y + blockDim.y * gridDim.y;
  int neighbours = neighbour_count(board_in, x, y);
  int cell = board_in[x + y];
  board_out[x + y] = step_cell(cell, neighbours);
}
