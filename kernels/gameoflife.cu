__device__ int neighbour_count(int *board, int idx) {
  int count = 0;
  for (int i = -1; i <= 1; i++) {
    for (int j = -1; j <= 1; j++) {
      int newIdx = idx + i + (j * ($MATRIX_WIDTH + 2));
      count += board[newIdx];
    }
  }
  count -= board[idx];
  return count;
}

// Just for testing
__global__ void neighbour_count_test(int *out, int *board, int idx) {
  *out = neighbour_count(board, idx);
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
  int x = (threadIdx.x + 1); //+ gridDim.x;
  int y = (threadIdx.y + 1) * ($MATRIX_WIDTH + 2);
  int idx = x + y;
  int neighbours = neighbour_count(board_in, idx);
  int cell = board_in[idx];
  board_out[idx] = step_cell(cell, neighbours);
}
