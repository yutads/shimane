import numpy as np
import pandas as pd
import random
import math
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import torch
import torch.nn as nn

import yaml
from attrdict import AttrDict

from module import PositionalEncoding
from module import Model
from module import MyDataset
from module import EarlyStopping

from my_def import transform_float_read
from my_def import make_info_data
from my_def import make_data
from my_def import compute_loss
from my_def import train_step
from my_def import val_step
from my_def import test_step
from my_def import seed_worker
from my_def import train_model

device = torch.device('cuda:0')# if torch.cuda.is_available() else 'cpu'

lis = list(sys.argv)
s = lis[1]

import datetime
dt_now = datetime.datetime.now()
print(dt_now.strftime('%Y年%m月%d日 %H:%M:%S'))
print(f'start {s}')

#3分ぐらいかかる
df = pd.read_csv(f'/gs/hs0/tga-akamalab/shimane_data/roi_timeseries/timeseries_{s}.csv')
df_dynamic = pd.read_csv(f'/gs/hs0/tga-akamalab/shimane_data/dynamic_FC/dynamic_{s}.csv',
                             dtype = transform_float_read(df))
df_subjects_info = pd.read_csv('../02_data_analysis/subjects_info/subjects_info.csv')

data_label = make_info_data()
data = make_data(df)
data_dynamic = make_data(df_dynamic)

age_mean = df_subjects_info.Age.mean()
age_std = df_subjects_info.Age.std()

n_splits = 5

with open('param.yaml') as f:
    config = yaml.safe_load(f)# config is dict
    cfg = AttrDict(config)

param_transformer_ROI1 = cfg.param_transformer_ROI1
param_transformer_ROI2 = cfg.param_transformer_ROI2
param_transformer_dFC1 = cfg.param_transformer_dFC1
param_transformer_dFC2 = cfg.param_transformer_dFC2

def nested_predict(data,data_label,train_model,param,title,batch_size):
    test_pred_all  = []
    test_pred_corrected_all  = []
    test_label_all = []
    test_idx_all   = []
    for fold_idx, (train_idx, test_idx) in enumerate(fold.split(data)):
        print(f'Nested {fold_idx + 1} Fold')
        test_pred,test_pred_corrected,test_label = train_model(data = data,
                                                               data_label = data_label,
                                                               train_idx  = train_idx,
                                                               test_idx   = test_idx,
                                                               n_splits   = n_splits,
                                                               epochs     = param['epochs'],
                                                               d_model    = param['d_mode'],
                                                               hidden_dim = param['hidden_dim'],
                                                               nhead      = param['nhead'],
                                                               hidden_dim_transformer = param['hidden_dim_transformer'],
                                                               pos_drop   = param['pos_drop'],
                                                               trans_drop = param['trans_drop'],
                                                               fc_drop    = param['fc_drop'],
                                                               title      = title,
                                                               atlas_name = s,
                                                               batch_size = batch_size
                                                              )
        
        test_pred_all.append(test_pred)
        test_pred_corrected_all.append(test_pred_corrected)
        test_label_all.append(test_label)
        test_idx_all.append(test_idx)
    
    return test_pred_all,test_pred_corrected_all,test_label_all,test_idx_all


dt_now = datetime.datetime.now()
with open("output/age_predict_transformer.txt", 'a') as f:
    f.write(dt_now.strftime('%Y年%m月%d日 %H:%M:%S')+ f'\nROI1 start {s}\n')

fold = KFold(n_splits=n_splits,shuffle=False)
test_pred_1,\
test_pred_corrected_1,\
test_label_1,\
test_idx_1 = nested_predict(data,data_label,
                            train_model = train_model,
                            param = param_transformer_ROI1,
                            title=f'ROI {s} model1',
                            batch_size = 16
                           )

dt_now = datetime.datetime.now()
with open("output/age_predict_transformer.txt", 'a') as f:
    f.write(dt_now.strftime('%Y年%m月%d日 %H:%M:%S')+ f'\nROI2 start {s}\n')

fold = KFold(n_splits=n_splits,shuffle=False)
test_pred_2,\
test_pred_corrected_2,\
test_label_2,\
test_idx_2 = nested_predict(data,data_label,
                            train_model = train_model,
                            param = param_transformer_ROI2,
                            title=f'ROI {s} model2',
                            batch_size = 16
                           )

test_pred_1 = np.array(test_pred_1, dtype=object)
test_pred_corrected_1 = np.array(test_pred_corrected_1, dtype=object)

test_pred_2 = np.array(test_pred_2, dtype=object)
test_pred_corrected_2 = np.array(test_pred_corrected_2, dtype=object)

np.savez(f'../02_data_analysis/temp/{s}_1',
         test_pred_1,
         test_pred_corrected_1,
         test_label_1,
         test_idx_1)

np.savez(f'../02_data_analysis/temp/{s}_2',
         test_pred_2,
         test_pred_corrected_2,
         test_label_2,
         test_idx_2)


dt_now = datetime.datetime.now()
with open("output/age_predict_transformer.txt", 'a') as f:
    f.write(dt_now.strftime('%Y年%m月%d日 %H:%M:%S')+ f'\ndFC1 start {s}\n')

fold = KFold(n_splits=n_splits,shuffle=False)
test_pred_dynamic_1,\
test_pred_dynamic_corrected_1,\
test_label_dynamic_1,\
test_idx_dynamic_1 = nested_predict(data_dynamic,data_label,
                                    train_model = train_model,
                                    param = param_transformer_dFC1,
                                    title=f'DynamicFC {s} model1',
                                    batch_size = 8
                                   )

dt_now = datetime.datetime.now()
with open("output/age_predict_transformer.txt", 'a') as f:
    f.write(dt_now.strftime('%Y年%m月%d日 %H:%M:%S')+ f'\ndFC2 start {s}\n')

fold = KFold(n_splits=n_splits,shuffle=False)
test_pred_dynamic_2,\
test_pred_dynamic_corrected_2,\
test_label_dynamic_2,\
test_idx_dynamic_2 = nested_predict(data_dynamic,data_label,
                                    train_model = train_model,
                                    param = param_transformer_dFC2,
                                    title=f'DynamicFC {s} model2',
                                    batch_size = 8
                                   )

    
test_pred_dynamic_1 = np.array(test_pred_dynamic_1, dtype=object)
test_pred_dynamic_corrected_1 = np.array(test_pred_dynamic_corrected_1, dtype=object)

test_pred_dynamic_2 = np.array(test_pred_dynamic_2, dtype=object)
test_pred_dynamic_corrected_2 = np.array(test_pred_dynamic_corrected_2, dtype=object)

np.savez(f'../02_data_analysis/temp/dynamic_{s}_1',
         test_pred_dynamic_1,
         test_pred_dynamic_corrected_1,
         test_label_dynamic_1,
         test_idx_dynamic_1)
np.savez(f'../02_data_analysis/temp/dynamic_{s}_2',
         test_pred_dynamic_2,
         test_pred_dynamic_corrected_2,
         test_label_dynamic_2,
         test_idx_dynamic_2)

np.savez(f'../02_data_analysis/temp/{s}_dynamic_1',
         test_pred_dynamic_1,
         test_pred_dynamic_corrected_1,
         test_label_dynamic_1,
         test_idx_dynamic_1)
np.savez(f'../02_data_analysis/temp/{s}_dynamic_2',
         test_pred_dynamic_2,
         test_pred_dynamic_corrected_2,
         test_label_dynamic_2,
         test_idx_dynamic_2)