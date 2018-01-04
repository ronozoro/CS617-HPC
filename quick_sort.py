import itertools
import random
from random import randrange

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()
arr = random.sample(range(100), 10)


def partition(lst, start, end, pivot):
    lst[pivot], lst[end] = lst[end], lst[pivot]
    store_index = start
    for i in range(start, end):
        if lst[i] < lst[end]:
            lst[i], lst[store_index] = lst[store_index], lst[i]
            store_index += 1
    lst[store_index], lst[end] = lst[end], lst[store_index]
    return store_index


def quick_sort(lst, start, end):
    if start >= end:
        return lst
    pivot = randrange(start, end + 1)
    new_pivot = partition(lst, start, end, pivot)
    quick_sort(lst, start, new_pivot - 1)
    quick_sort(lst, new_pivot + 1, end)
    return lst


def chunk(xs, n):
    L = len(xs)
    assert 0 < n <= L
    s, r = divmod(L, n)
    chunks = [xs[p:p + s] for p in range(0, L, s)]
    chunks[n - 1:] = [xs[-r - s:]]
    return chunks


if rank == 0:
    wt = MPI.Wtime()
    print('Our list is going ot be scattered : ', arr)
    arr = chunk(arr, size)
arr = comm.scatter(arr, root=0)
print("Rank : ", rank, " processor : ", name, ' Has list: ', arr)
quick_time = MPI.Wtime()
quick_sort(arr, 0, len(arr) - 1)
print("Rank : ", rank, " processor : ", name, ' After sort list: ', arr,
      " in time : %6.6fs " % (MPI.Wtime() - quick_time))
new_arr = comm.gather(arr, root=0)
if rank == 0:
    global_arr = list(itertools.chain.from_iterable(new_arr))
    sorted = quick_sort(global_arr, 0, len(global_arr) - 1)
    wt1 = MPI.Wtime() - wt
    print("Sorted :", sorted, " in time %6.6fs" % wt1)
