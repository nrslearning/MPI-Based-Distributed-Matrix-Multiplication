from mpi4py import MPI
import numpy as np
import time

def serial_matrix_multiply(A, B):
    return np.dot(A, B)

def parallel_matrix_multiply(comm, A, B, N):
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Determine the number of rows per process
    rows_per_proc = N // size
    extra = N % size

    # Distribute rows unevenly if not divisible
    if rank < extra:
        local_rows = rows_per_proc + 1
        start_row = rank * local_rows
    else:
        local_rows = rows_per_proc
        start_row = rank * local_rows + extra

    end_row = start_row + local_rows

    # Scatter rows of matrix A
    local_A = A[start_row:end_row, :] if rank == 0 else np.empty((local_rows, N), dtype=np.float64)

    if rank != 0:
        B = np.empty((N, N), dtype=np.float64)

    comm.Bcast(B, root=0)
    comm.Scatterv([A, get_counts(N, size), get_displacements(N, size), MPI.DOUBLE], local_A, root=0)

    # Local multiplication
    local_C = np.dot(local_A, B)

    # Gather results
    C = None
    if rank == 0:
        C = np.empty((N, N), dtype=np.float64)
    comm.Gatherv(local_C, [C, get_counts(N, size, result=True), get_displacements(N, size, result=True), MPI.DOUBLE], root=0)

    return C

def get_counts(N, size, result=False):
    base = N // size
    extra = N % size
    if result:
        return [(base + 1) * N if i < extra else base * N for i in range(size)]
    return [(base + 1) * N if i < extra else base * N for i in range(size)]

def get_displacements(N, size, result=False):
    base = N // size
    extra = N % size
    displ = [0]
    for i in range(1, size):
        prev = base + 1 if i - 1 < extra else base
        displ.append(displ[-1] + prev * N)
    return displ

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"####### Rank{rank} Size{size}",flush=True)

    N = 512  # Size of NxN matrix

    if rank == 0:
        print(f"Running with {size} processes on {N}x{N} matrix")
        A = np.random.rand(N, N)
        B = np.random.rand(N, N)
        start = time.time()
        C_serial = serial_matrix_multiply(A, B)
        serial_time = time.time() - start
        print(f"Serial execution time: {serial_time:.4f} seconds")
    else:
        A = None
        B = None

    comm.Barrier()
    start = MPI.Wtime()
    C_parallel = parallel_matrix_multiply(comm, A, B, N)
    end = MPI.Wtime()

    if rank == 0:
        parallel_time = end - start
        print(f"Parallel execution time with {size} processes: {parallel_time:.4f} seconds",flush=True)
        print(f"Speedup: {serial_time / parallel_time:.2f}x",flush=True)

if __name__ == "__main__":
    main()
