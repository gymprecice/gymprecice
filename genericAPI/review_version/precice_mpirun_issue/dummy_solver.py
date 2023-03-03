## a solver that we would like to run in parallel within our adapter

from mpi4py import MPI

comm = MPI.COMM_WORLD

print("%d of %d" % (comm.Get_rank(), comm.Get_size()))
