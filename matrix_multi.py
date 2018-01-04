import numpy as np
from mpi4py import MPI

N = 20
TaskMaster = 0

a = np.random.randint(100, size=(N, N))
b = np.random.randint(100, size=(N, N))
comm = MPI.COMM_WORLD
worldSize = comm.Get_size()
rank = comm.Get_rank()
processorName = MPI.Get_processor_name()

print("Process %d started.\n" % (rank + 1))
print("Running from processor %s, rank %d out of %d processors.\n" % (processorName, rank + 1, worldSize))

if (worldSize == 1):
    slice = int(N)
else:
    slice = int(N / (worldSize - 1))
comm.Barrier()

if rank == TaskMaster:
    print("Start")
    print('A Matrix ' + str(N) + '*' + str(N), a)
    print('B Matrix ' + str(N) + '*' + str(N), b)
    for i in range(1, worldSize):
        offset = (i - 1) * slice  # 0, 10, 20
        row = a[0, :]
        comm.send(offset, dest=i, tag=i)
        comm.send(row, dest=i, tag=i)
        for j in range(0, slice):
            comm.send(a[j + offset, :], dest=i, tag=j + offset)
    print("All sent to workers.\n")

comm.Barrier()

if rank != TaskMaster:
    slice = int(slice)
    print("Data Received from process %d.\n" % (rank))
    offset = comm.recv(source=0, tag=rank)
    recv_data = comm.recv(source=0, tag=rank)
    for j in range(1, slice):
        c = comm.recv(source=0, tag=j + offset)
        recv_data = np.vstack((recv_data, c))
    print("Start Calculation from process %d.\n" % (rank))

    t_start = MPI.Wtime()
    for i in range(0, slice):
        res = np.zeros(shape=(N))
        if (slice == 1):
            r = recv_data
        else:
            r = recv_data[i, :]
        ai = 0
        for j in range(0, N):
            q = b[:, j]  # get the column we want
            for x in range(0, N):
                res[j] = res[j] + (r[x] * q[x])
            ai = ai + 1
        if (i > 0):
            send = np.vstack((send, res))
        else:
            send = res
    t_diff = MPI.Wtime() - t_start

    print("Process %d finished in %5.4fs.\n" % (rank, t_diff))
    print("Sending results to Master %d bytes.\n" % (send.nbytes))
    comm.Send([send, MPI.FLOAT], dest=0, tag=rank)  # 1, 12, 23

comm.Barrier()

if rank == TaskMaster:
    res1 = np.zeros(shape=(slice, N))
    comm.Recv([res1, MPI.FLOAT], source=1, tag=1)
    kl = np.vstack((res1))
    for i in range(2, worldSize):
        resx = np.zeros(shape=(slice, N))
        comm.Recv([resx, MPI.FLOAT], source=i, tag=i)
        print("Received response from %d.\n" % (i))
        kl = np.vstack((kl, resx))
    print("End")
    print("Result A*B.\n")
    print(kl)

comm.Barrier()
