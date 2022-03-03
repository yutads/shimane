#!/bin/sh
#$ -cwd
#$ -e output/zzz3.err
#$ -o output/zzz3.out
#$ -l f_node=2
#$ -l h_rt=0:10:00
#$ -N zzz3
#$ -V

. /etc/profile.d/modules.sh
module load cuda
module load intel
module load jupyterlab
module load gcc/8.3.0 cuda openmpi nccl/2.4.2

HOROVOD_GPU_OPERATIONS=NCCL
HOROVOD_NCCL_HOME=/apps/t3/sles12sp2/free/nccl/2.4.2/gcc4.8.5/cuda9.2 python3 -m pip install --user --no-cache-dir horovod

mpirun -npernode 4 -np 8 \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python zzz3.py