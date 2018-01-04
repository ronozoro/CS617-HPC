import itertools
import random

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()
arr = random.sample(range(100), 50)


def getMinMax(List):
    Minmax = [0, 0]
    if len(List) == 1:
        Minmax[1] = List[0]
        Minmax[0] = List[0]
        return Minmax

    if List[0] > List[1]:
        Minmax[1] = List[0]
        Minmax[0] = List[1]
    else:
        Minmax[1] = List[1]
        Minmax[0] = List[0]

    for i in range(2, len(List)):
        if List[i] > Minmax[1]:
            Minmax[1] = List[i]
        elif List[i] < Minmax[0]:
            Minmax[0] = List[i]

    return Minmax


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
arr = getMinMax(arr)
print("Rank : ", rank, ' After GetMinMax: ', arr)
new_arr = comm.gather(arr, root=0)
if rank == 0:
    global_arr = list(itertools.chain.from_iterable(new_arr))
    minmax = getMinMax(global_arr)
    wt1 = MPI.Wtime() - wt
    print("Total Time MinMax :", minmax, " in time %6.6fs" % wt1)
