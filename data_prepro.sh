#!/bin/sh
#$ -cwd
#$ -e output/data_prepro.err
#$ -o output/data_prepro.out
#$ -l f_node=1
#$ -l h_rt=23:50:00
#$ -N data_prepro

. /etc/profile.d/modules.sh
module load cuda
module load intel
module load jupyterlab

#aal
#ho
#msdl
#Yeo
#destrieux
#Dosenbach
#Power
#seitzman
#craddock

python data_prepro.py aal
python data_prepro.py ho
python data_prepro.py msdl
python data_prepro.py Yeo
python data_prepro.py destrieux
python data_prepro.py Dosenbach
python data_prepro.py Power
python data_prepro.py seitzman
#python data_prepro.py craddock