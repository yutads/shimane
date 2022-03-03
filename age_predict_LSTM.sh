#!/bin/sh
#$ -cwd
#$ -e output/age_predict_LSTM.err
#$ -o output/age_predict_LSTM.out
#$ -l f_node=1
# -l h_rt=00:10:00
#$ -l h_rt=23:50:00
#$ -N age_predict_LSTM

. /etc/profile.d/modules.sh
module load cuda
module load intel
module load jupyterlab

#craddock
#msdl
#Yeo
#aal
#ho
#destrieux
#Dosenbach
#Power
#seitzman

#python age_predict_LSTM.py msdl
#python age_predict_LSTM.py Yeo

#python age_predict_LSTM.py ho&
#python age_predict_LSTM.py aal&
#wait

#python age_predict_LSTM.py destrieux&
#python age_predict_LSTM.py Dosenbach&
#wait

python age_predict_LSTM.py Dosenbach

#python age_predict_LSTM.py Power
#python age_predict_LSTM.py seitzman

#python age_predict_LSTM.py craddock