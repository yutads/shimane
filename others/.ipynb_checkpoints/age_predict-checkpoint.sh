#!/bin/sh
#$ -cwd
#$ -e output/age_predict.err
#$ -o output/age_predict.out
#$ -l f_node=1
#$-l h_rt=23:50:00
#$ -N age_predict

. /etc/profile.d/modules.sh
module load cuda
module load intel
module load jupyterlab

python GPU.py

#jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute 02_age_predict.ipynb &
#jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute 02_age_predict_multi_smoothing.ipynb &

jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute 02_gender_classification.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute 02_age_predict_lstm.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute 02_age_predict_lstm_reverse.ipynb