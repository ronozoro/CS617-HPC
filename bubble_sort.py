import itertools
import random

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()
arr = random.sample(range(100), 10)


def bubblesort(L):
    for i in range(len(L)):
        swapped = False
        for j in range(len(L) - i - 1):
            if L[j] > L[j + 1]:
                L[j], L[j + 1] = L[j + 1], L[j]
                swapped = True
        if not swapped:
            return L
    return L


def chunk(xs, n):
    L = len(xs)
    assert 0 < n <= L
    s, r = divmod(L, n)
    chunks = [xs[p:p + s] for p in range(0, L, s)]
    chunks[n - 1:] = [xs[-r - s:]]
    return chunks


if rank == 0:
    wt = MPI.Wtime()
    print('Our list is going to be scattered : ', arr)
    arr = chunk(arr, size)
arr = comm.scatter(arr, root=0)
print("Rank : ", rank, ' Has list: ', arr)
arr = bubblesort(arr)
print("Rank : ", rank, ' After sort list: ', arr)
new_arr = comm.gather(arr, root=0)
if rank == 0:
    global_arr = list(itertools.chain.from_iterable(new_arr))
    sorted = bubblesort(global_arr)
    wt1 = MPI.Wtime() - wt
    print("Sorted :", sorted, " in time %6.6fs" % wt1)
