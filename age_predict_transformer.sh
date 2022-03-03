#!/bin/sh
#$ -cwd
#$ -e output/age_predict_transformer.err
#$ -o output/age_predict_transformer.out
#$ -l f_node=1
# -l h_rt=00:10:00
#$ -l h_rt=23:50:00
#$ -N age_predict_transformer

. /etc/profile.d/modules.sh
module load cuda
module load intel
module load jupyterlab

python age_predict_transformer.py msdl
python age_predict_transformer.py Yeo
python age_predict_transformer.py ho
python age_predict_transformer.py aal
python age_predict_transformer.py destrieux
python age_predict_transformer.py Dosenbach
python age_predict_transformer.py Power
python age_predict_transformer.py seitzman
python age_predict_transformer.py craddock