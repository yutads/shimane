#!/bin/sh
#$ -cwd
#$ -e output/age_predict_transformer_multi.err
#$ -o output/age_predict_transformer_multi.out
#$ -l f_node=1
# -l h_rt=00:10:00
#$ -l h_rt=23:50:00
#$ -N age_predict_transformer_multi

. /etc/profile.d/modules.sh
module load cuda
module load intel
module load jupyterlab

python age_predict_transformer_multi.py msdl
python age_predict_transformer_multi.py Yeo
python age_predict_transformer_multi.py ho
python age_predict_transformer_multi.py aal
python age_predict_transformer_multi.py destrieux
python age_predict_transformer_multi.py Dosenbach
python age_predict_transformer_multi.py Power
python age_predict_transformer_multi.py seitzman
python age_predict_transformer_multi.py craddock