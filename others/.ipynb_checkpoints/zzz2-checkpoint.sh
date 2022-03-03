#!/bin/sh
#$ -cwd
#$ -e output/zzz2.err
#$ -o output/zzz2.out
#$ -l f_node=2
#$ -l h_rt=0:10:00
#$ -N zzz2
#$ -V

. /etc/profile.d/modules.sh
module load cuda
module load intel
module load jupyterlab
module load gcc/8.3.0 cuda openmpi

#echo $PATH
nodes=( $(cat $PE_HOSTFILE |awk '{print $1":"$2}') )
#echo ${nodes[@]}
#https://horovod.readthedocs.io/en/stable/mpi_include.html
#horovodrun -npernode 4 -np 8 python zzz2.py
#horovodrun -np 56 -H ${nodes[0]}, ${nodes[0]} python zzz2.py
horovodrun -np 4 python zzz2.py
horovodrun -np 8 python zzz2.py
#mpirun -npernode 4 -np 8 \
#    -bind-to none -map-by slot \
#    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
#    -mca pml ob1 -mca btl ^openib \
#    python zzz2.py
    
python zzz2.py

mpirun -npernode 4 -np 8 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python zzz2.py