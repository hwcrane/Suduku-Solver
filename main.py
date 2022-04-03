import numpy as np
import time
import numba


@numba.jit(nopython=True)
def check(puzzle: np.array, rowNum: int, columnNum: int, n: int):
    # Row element is in
    r = puzzle[rowNum, :]

    # Column element is in
    c = puzzle[:, columnNum]

    # subgrid elment is in
    x, y = (rowNum//3) * 3, (columnNum//3) * 3
    s = puzzle[x: x+3, y: y+3]

    # Check that n does not occur in any of them
    return np.sum(r == n) == 0 and np.sum(c == n) == 0 and np.sum(s == n) == 0


@numba.jit(nopython=True)
def solve(grid: np.array):
    # Get the indexes of the unsolved elements
    rows, columns = np.where(grid == 0)

    # Check if puzzle has been solved
    if len(rows) == 0:
        return True

    # Loop over the unsolved indexes
    for i, row in enumerate(rows):
        col = columns[i]
        for n in range(1, 10):

            # Check if solution is possible
            if check(grid, row, col, n):

                # Assign element to posible solution
                grid[row, col] = n

                # Reqursive call
                if solve(grid):
                    return True

                # if solution is later shown to be not possible,
                # set element back to 0
                grid[row, col] = 0

        # If none of the solutions were possible, returns false
        return False


puzzle = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 5, 4, 0, 0, 0, 7, 9],
    [0, 9, 0, 0, 0, 0, 6, 4, 5],
    [0, 4, 0, 0, 0, 2, 1, 0, 0],
    [0, 0, 0, 0, 3, 0, 0, 2, 0],
    [0, 0, 0, 1, 0, 0, 3, 5, 8],
    [0, 0, 9, 7, 0, 5, 0, 0, 0],
    [0, 1, 4, 0, 9, 6, 0, 0, 0],
    [0, 6, 2, 0, 0, 3, 0, 0, 0]
])

start = time.time()
solve(puzzle)
end = time.time() - start
print(puzzle)
print(f'Time taken: {round(end, 5)}s')
