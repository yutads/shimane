#!/bin/sh
#$ -cwd
#$ -e aaa.err
#$ -o aaa.out
#$ -l f_node=1
#$ -l h_rt=0:10:00
#$ -N aaa

. /etc/profile.d/modules.sh
module load cuda
module load intel
module load jupyterlab
module load gcc/8.3.0 cuda openmpi

#python3 -m pip install --user horovod