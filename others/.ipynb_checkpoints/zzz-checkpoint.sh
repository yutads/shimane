#!/bin/sh
#$ -cwd
#$ -e output/zzz.err
#$ -o output/zzz.out
#$ -l f_node=1
#$ -l h_rt=0:10:00
#$ -N zzz

. /etc/profile.d/modules.sh
module load cuda
module load intel
module load jupyterlab

python zzz.py