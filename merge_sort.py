import itertools
import random

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()
arr = random.sample(range(100), 10)


def merge(left, right):
    if not len(left) or not len(right):
        return left or right

    result = []
    i, j = 0, 0
    while (len(result) < len(left) + len(right)):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
        if i == len(left) or j == len(right):
            result.extend(left[i:] or right[j:])
            break

    return result


def mergesort(list):
    if len(list) < 2:
        return list
    middle = int(round(len(list) / 2))
    left = mergesort(list[:middle])
    right = mergesort(list[middle:])
    return merge(left, right)


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

print("Rank : ", rank, ' Has list: ', arr, "processor : ", name)
merge_time = MPI.Wtime()
mergesort(arr)
print("Rank : ", rank, " processor : ", name, ' After sort list: ', arr,
      " in time : %6.6fs " % (MPI.Wtime() - merge_time))
new_arr = comm.gather(arr, root=0)
if rank == 0:
    global_arr = list(itertools.chain.from_iterable(new_arr))
    sorted = mergesort(global_arr)
    wt1 = MPI.Wtime() - wt
    print("Sorted :", sorted, " in total time %6.6fs" % wt1)
