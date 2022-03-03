#!/bin/sh
#$ -cwd
#$ -e output/age_predict_transformer_single.err
#$ -o output/age_predict_transformer_single.out
#$ -l f_node=1
# -l h_rt=00:10:00
#$ -l h_rt=23:50:00
#$ -N age_predict_transformer_single

. /etc/profile.d/modules.sh
module load cuda
module load intel
module load jupyterlab

python age_predict_transformer_single.py msdl
python age_predict_transformer_single.py Yeo
python age_predict_transformer_single.py ho
python age_predict_transformer_single.py aal
python age_predict_transformer_single.py destrieux
python age_predict_transformer_single.py Dosenbach
python age_predict_transformer_single.py Power
python age_predict_transformer_single.py seitzman
python age_predict_transformer_single.py craddock