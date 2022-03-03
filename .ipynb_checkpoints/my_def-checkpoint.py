import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import torch
import torch.nn as nn

from module import PositionalEncoding
from module import Model
from module import Model_LSTM
from module import Model_multi
from module import MyDataset
from module import EarlyStopping

df_subjects_info = pd.read_csv('../02_data_analysis/subjects_info/subjects_info.csv')
device = torch.device("cuda:0")# if torch.cuda.is_available() else 'cpu'
age_mean = df_subjects_info.Age.mean()
age_std = df_subjects_info.Age.std()

import datetime
import sys

def transform_float_read(df):
    extra = 2
    lis_col = list(df.columns)[extra:]#ROI名
    lis_col_new = list(df.columns)[:extra]#dynamic_FCのcolumns
    for m in range(len(lis_col)):
        for n in range(m+1,len(lis_col)):
            lis_col_new.append(lis_col[m] + '_' + lis_col[n])
    keys = lis_col_new
    values = [np.int64,np.object]
    values += [np.float16]*len(keys)
    d = dict(zip(keys, values))
    return d
    
def make_info_data(df = df_subjects_info,col = ['Age','Sex']):
    subID = np.array(df.subID)
    subjects_info_list = []
    for ID in subID:
        data = np.array(df[df.subID == ID][col])[0]
        if 'Sex' in col:
            idx_sex = col.index('Sex')
            if data[idx_sex] == '男':
                data[idx_sex] = 0
            else:
                data[idx_sex] = 1
        
        subjects_info_list.append(data)
    subjects_info = np.array(subjects_info_list)
    
    if 'Age' in col:
        idx_age = col.index('Age')
    for i in range(len(col)):
        if i == idx_age:
            subjects_info[:,i] = (subjects_info[:,i] - \
                                  np.mean(subjects_info[:,i]))/np.std(subjects_info[:,i])
    return subjects_info.astype(np.float)

def make_data(df, subID = df_subjects_info.subID ,remove_list = ['time','subID']):
    data_list = []
    df_cols = list(df.columns)
    for r in remove_list:
        try:
            df_cols.remove(r)
        except:
            pass
    for ID in df_subjects_info.subID:
        data = df[df.subID == ID][df_cols]
        data = np.array(data)
        data_list.append(data)
    data = np.array(data_list)
    return data

def compute_loss(y, t):
    criterion = nn.MSELoss()
    return criterion(y, t)

def compute_loss_age(y, t):
    criterion = nn.MSELoss()
    return criterion(y, t)

def compute_loss_sex(y, t):
    criterion = nn.BCELoss()
    return criterion(y, t)

def train_step(x,t,model,optimizer):
    model.train()
    preds = model(x)
    loss = compute_loss(preds, t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, preds

def val_step(x,t,model):
    model.eval()
    preds = model(x)
    loss = compute_loss(preds, t)    
    return loss, preds

def test_step(x,model):
    model.eval()
    preds = model(x)
    return preds

def seed_worker(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def train_step_single(x,t,model,optimizer):
    model.train()
    preds_age,preds_sex = model(x)
    
    age = t[:,0:1].detach().clone()
    sex = t[:,1:].detach().clone()
    
    sex_0 = sex == 0
    sex_1 = sex == 1
    
    sex[sex_0] = 0.3
    sex[sex_1] = 0.7
    
    loss_age = compute_loss_age(preds_age, age)
    loss_sex = compute_loss_sex(preds_sex, sex)
    loss = 0 * loss_age + 1 * loss_sex

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, preds_age, preds_sex

def val_step_single(x,t,model):
    model.eval()
    preds_age,preds_sex = model(x)
    
    age = t[:,0:1].detach().clone()
    sex = t[:,1:].detach().clone()
    
    #sex_0 = sex == 0
    #sex_1 = sex == 1
    
    #sex[sex_0] = 0.3
    #sex[sex_1] = 0.7
    
    loss_age = compute_loss_age(preds_age, age)
    loss_sex = compute_loss_sex(preds_sex, sex)
    loss = 0 * loss_age + 1 * loss_sex
    
    return loss, preds_age, preds_sex

def test_step_single(x,model):
    model.eval()
    preds_age,preds_sex = model(x)
    return preds_age, preds_sex

def train_step_multi(x,t,model,optimizer):
    model.train()
    preds_age,preds_sex = model(x)
    
    age = t[:,0:1].detach().clone()
    sex = t[:,1:].detach().clone()
    
    sex_0 = sex == 0
    sex_1 = sex == 1
    
    sex[sex_0] = 0.3
    sex[sex_1] = 0.7
    
    loss_age = compute_loss_age(preds_age, age)
    loss_sex = compute_loss_sex(preds_sex, sex)
    loss = (4/5) * loss_age + (1/5) * loss_sex

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, preds_age, preds_sex

def val_step_multi(x,t,model):
    model.eval()
    preds_age,preds_sex = model(x)
    age = t[:,0:1].detach().clone()
    sex = t[:,1:].detach().clone()
    
    #sex_0 = sex == 0
    #sex_1 = sex == 1
    
    #sex[sex_0] = 0.3
    #sex[sex_1] = 0.7
    
    loss_age = compute_loss_age(preds_age, age)
    loss_sex = compute_loss_sex(preds_sex, sex)
    loss = (4/5) * loss_age + (1/5) * loss_sex
    
    return loss, preds_age, preds_sex

def test_step_multi(x,model):
    model.eval()
    preds_age,preds_sex = model(x)
    return preds_age, preds_sex
    
def train_model(data,
                data_label,
                train_idx,
                test_idx,
                n_splits,
                atlas_name,
                epochs     = 750,
                d_model    = 128,
                hidden_dim = 512,
                nhead      = 32,
                hidden_dim_transformer = 2,
                pos_drop    = 0.1,
                trans_drop = 0.1,
                fc_drop = 0.1,
                title = '',
                batch_size = 32,
                ):
    
    epochs = epochs
    seed_worker()
    data_label = np.array([data_label[:,0]]).T
    
    model_path = f'../02_data_analysis/model/model_{atlas_name}.pth'
    
    train_all = data[train_idx]
    test  = data[test_idx]
    
    train_label_all = data_label[train_idx]
    test_label  = data_label[test_idx]
    
    input_dim  = data.shape[2]
    time_len   = data.shape[1]
    output_dim = data_label.shape[1]
    d_model    = d_model
    hidden_dim = hidden_dim
    nhead      = nhead
    hidden_dim_transformer = hidden_dim_transformer
    pos_drop    = pos_drop
    trans_drop = trans_drop
    fc_drop = fc_drop

    if torch.cuda.device_count() > 1:
        batch_size = batch_size * torch.cuda.device_count()
    else:
        batch_size = batch_size
        
    test_dataset = MyDataset(test)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  num_workers=8,
                                                  persistent_workers=True,
                                                 )
    
    kf = KFold(n_splits=n_splits,shuffle=False)
    test_pred_all = []
    test_pred_all_corrected = []
    dt_now = datetime.datetime.now()
    
    for n_fold, (train_index, val_index) in enumerate(kf.split(train_all)):
        with open("output/age_predict_transformer.txt", 'a') as f:
            f.write(dt_now.strftime('    %Y年%m月%d日 %H:%M:%S')+ f'\n    n_fold:{n_fold}\n')
        save_epochs = 0
        train_loss_plot = []
        val_loss_plot = []
        
        train, train_label = train_all[train_index], train_label_all[train_index]
        val,   val_label   = train_all[val_index],   train_label_all[val_index]
        
        train_dataset    = MyDataset(train,train_label)
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=True,
                                                       num_workers=8,
                                                       persistent_workers=True,
                                                      )
        
        val_dataset      = MyDataset(val,val_label)
        val_dataloader   = torch.utils.data.DataLoader(val_dataset,
                                                       batch_size=batch_size,
                                                       num_workers=8,
                                                       persistent_workers=True,
                                                      )
        
        model = Model(input_dim = input_dim,
                      d_model =d_model,
                      hidden_dim = hidden_dim,
                      time_len = time_len,
                      nhead = nhead,
                      output_dim = output_dim,
                      hidden_dim_transformer = hidden_dim_transformer,
                      pos_drop = pos_drop,
                      trans_drop = trans_drop,
                      fc_drop = fc_drop
                     )
        
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)
        
        es = EarlyStopping(patience=200, verbose=1, mode='loss')
        best_val_loss = 1e5
        
        for epoch in range(epochs):
            train_loss = 0.
            val_loss = 0.

            for (x, t) in train_dataloader:
                x, t = x.to(device), t.to(device)
                loss, preds = train_step(x,t,model,optimizer)
                train_loss += loss.item()
            train_loss /= len(train_dataloader)
            
            for (x, t) in val_dataloader:
                x, t = x.to(device), t.to(device)
                loss, preds = val_step(x,t,model)
                val_loss += loss.item()
            val_loss /= len(val_dataloader)
            
            if (epoch+1) % 200 == 0:
                dt_now = datetime.datetime.now()
                print(f'Kfold: {n_fold+1} ::: epoch: {epoch+1}, loss: {train_loss}, val loss: {val_loss}')
                
            #early stopping
            if epoch <=200:
                pass
            elif es(val_loss):
                break

            # modelの保存
            if epoch <=200:
                pass
            elif val_loss < best_val_loss:
                save_epochs = epoch+1
                best_val_loss = val_loss
                torch.save(model.module.state_dict(), model_path)

            train_loss_plot.append(train_loss)
            val_loss_plot.append(val_loss)

        model = Model(input_dim = input_dim,
                      d_model =d_model,
                      hidden_dim = hidden_dim,
                      time_len = time_len,
                      nhead = nhead,
                      output_dim = output_dim,
                      hidden_dim_transformer = hidden_dim_transformer,
                      pos_drop = pos_drop,
                      trans_drop = trans_drop,
                      fc_drop = fc_drop
                     )
        
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        
        train_loss_plot = np.array(train_loss_plot)
        val_loss_plot   = np.array(val_loss_plot)
        x               = np.linspace(0, len(train_loss_plot), len(train_loss_plot))
        
        train_pred = []
        train_label = []
        
        for (x, t) in train_dataloader:
            x, t = x.to(device), t.to(device)
            loss, preds = val_step(x,t,model)
            train_pred.append(preds)
            train_label.append(t)
        
        val_pred = []
        for (x, t) in val_dataloader:
            x, t = x.to(device), t.to(device)
            loss, preds = val_step(x,t,model)
            val_pred.append(preds)
        
        train_pred = torch.cat(train_pred)
        train_pred = train_pred.to('cpu').detach().numpy().copy()
        
        train_label = torch.cat(train_label)
        train_label = train_label.to('cpu').detach().numpy().copy()
        
        val_pred = torch.cat(val_pred)
        val_pred = val_pred.to('cpu').detach().numpy().copy()

        lr = LinearRegression()
        lr.fit(val_label*age_std+age_mean,val_pred[:,0]*age_std+age_mean)
                
        test_pred = []
        for x in test_dataloader:
            x = x.to(device)
            preds = test_step(x,model)
            test_pred.append(preds)
        test_pred = torch.cat(test_pred)
        test_pred = test_pred.to('cpu').detach().numpy().copy()
        test_pred_corrected = (test_pred*age_std+age_mean - lr.intercept_)/ lr.coef_[0]
        test_pred_corrected = (test_pred_corrected-age_mean) / age_std
        if lr.coef_[0] > 0.1:
            test_pred_all.append(test_pred)
            test_pred_all_corrected.append(test_pred_corrected)
        
    return np.array(test_pred_all),np.array(test_pred_all_corrected),test_label

def train_model_LSTM(data,
                     data_label,
                     train_idx,
                     test_idx,
                     n_splits,
                     atlas_name,
                     bidirectional,
                     epochs     = 750,
                     hidden_dim = 128,
                     num_layers = 2,
                     fc_drop = 0.1,
                     title = '',
                     batch_size = 32
                    ):
    
    epochs = epochs
    seed_worker()
    data_label = np.array([data_label[:,0]]).T
    
    model_path = f'../02_data_analysis/model/model_{atlas_name}_LSTM.pth'
    
    train_all = data[train_idx]
    test  = data[test_idx]
    
    train_label_all = data_label[train_idx]
    test_label  = data_label[test_idx]
    
    input_dim  = data.shape[2]
    time_len   = data.shape[1]
    hidden_dim = hidden_dim
    num_layers = num_layers
    bidirectional = bidirectional
    fc_drop = fc_drop
    
    if torch.cuda.device_count() > 1:
        batch_size = batch_size * torch.cuda.device_count()
    else:
        batch_size = batch_size
    
    test_dataset = MyDataset(test)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  num_workers=8,
                                                  persistent_workers=True)
    
    kf = KFold(n_splits=n_splits,shuffle=False)
    test_pred_all = []
    test_pred_all_corrected = []
    dt_now = datetime.datetime.now()
    for n_fold, (train_index, val_index) in enumerate(kf.split(train_all)):
        with open("output/age_predict_LSTM.txt", 'a') as f:
            f.write(dt_now.strftime('    %Y年%m月%d日 %H:%M:%S')+ f'\n    n_fold:{n_fold}\n')
        save_epochs = 0
        train_loss_plot = []
        val_loss_plot = []
        
        train, train_label = train_all[train_index], train_label_all[train_index]
        val,   val_label   = train_all[val_index],   train_label_all[val_index]
        
        
        train_dataset    = MyDataset(train,train_label)
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=True,
                                                       num_workers=8,
                                                       persistent_workers=True)
    

        val_dataset      = MyDataset(val,val_label)
        val_dataloader   = torch.utils.data.DataLoader(val_dataset,
                                                       batch_size=batch_size,
                                                       num_workers=8,
                                                       persistent_workers=True)
        model = Model_LSTM(input_dim = input_dim,
                      hidden_dim = hidden_dim,
                      time_len = time_len,
                      num_layers = num_layers,
                      fc_drop = fc_drop,
                      bidirectional = bidirectional
                     )
        
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

        model.to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)
        
        es = EarlyStopping(patience=200, verbose=1, mode='loss')
        best_val_loss = 1e5
        
        for epoch in range(epochs):
            train_loss = 0.
            val_loss = 0.

            for (x, t) in train_dataloader:
                x, t = x.to(device), t.to(device)
                loss, preds = train_step(x,t,model,optimizer)
                train_loss += loss.item()
            train_loss /= len(train_dataloader)
            
            for (x, t) in val_dataloader:
                x, t = x.to(device), t.to(device)
                loss, preds = val_step(x,t,model)
                val_loss += loss.item()
            val_loss /= len(val_dataloader)
            
            if (epoch+1) % 200 == 0:
                print(f'Kfold: {n_fold+1} ::: epoch: {epoch+1}, loss: {train_loss}, val loss: {val_loss}')
                
            #early stopping
            if epoch <=200:
                pass
            elif es(val_loss):
                break

            # modelの保存
            if epoch <=200:
                pass
            elif val_loss < best_val_loss:
                save_epochs = epoch+1
                best_val_loss = val_loss
                torch.save(model.module.state_dict(), model_path)
        
            train_loss_plot.append(train_loss)
            val_loss_plot.append(val_loss)

        model = Model_LSTM(input_dim = input_dim,
                      hidden_dim = hidden_dim,
                      time_len = time_len,
                      num_layers = num_layers,
                      fc_drop = fc_drop,
                      bidirectional = bidirectional
                     )
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        
        train_loss_plot = np.array(train_loss_plot)
        val_loss_plot   = np.array(val_loss_plot)
        x               = np.linspace(0, len(train_loss_plot), len(train_loss_plot))
        
        train_pred = []
        train_label = []
        
        for (x, t) in train_dataloader:
            x, t = x.to(device), t.to(device)
            loss, preds = val_step(x,t,model)
            train_pred.append(preds)
            train_label.append(t)
        
        val_pred = []
        for (x, t) in val_dataloader:
            x, t = x.to(device), t.to(device)
            loss, preds = val_step(x,t,model)
            val_pred.append(preds)
        
        train_pred = torch.cat(train_pred)
        train_pred = train_pred.to('cpu').detach().numpy().copy()
        
        train_label = torch.cat(train_label)
        train_label = train_label.to('cpu').detach().numpy().copy()
        
        val_pred = torch.cat(val_pred)
        val_pred = val_pred.to('cpu').detach().numpy().copy()
        
        lr = LinearRegression()
        lr.fit(val_label*age_std+age_mean,val_pred[:,0]*age_std+age_mean)
        
        test_pred = []
        for x in test_dataloader:
            x = x.to(device)
            preds = test_step(x,model)
            test_pred.append(preds)
        test_pred = torch.cat(test_pred)
        test_pred = test_pred.to('cpu').detach().numpy().copy()
        test_pred_corrected = (test_pred*age_std+age_mean - lr.intercept_)/ lr.coef_[0]
        test_pred_corrected = (test_pred_corrected-age_mean) / age_std
        if lr.coef_[0] > 0.1:
            test_pred_all.append(test_pred)
            test_pred_all_corrected.append(test_pred_corrected)
        
    return np.array(test_pred_all),np.array(test_pred_all_corrected),test_label

def train_model_multi(data,
                      data_label,
                      train_idx,
                      test_idx,
                      n_splits,
                      atlas_name,
                      epochs     = 750,
                      d_model    = 128,
                      hidden_dim = 512,
                      nhead      = 32,
                      hidden_dim_transformer = 2,
                      pos_drop    = 0.1,
                      trans_drop = 0.1,
                      fc_drop = 0.1,
                      title = '',
                      batch_size = 32,
                      multi = True
                    ):
    
    epochs = epochs
    seed_worker()
    data_label = data_label[:,0:2]
    
    if multi:
        model_path = f'../02_data_analysis/model/model_multi_{atlas_name}_smooth.pth'
    else:
        model_path = f'../02_data_analysis/model/model_sigle_{atlas_name}_smooth.pth'

    train_all = data[train_idx]
    test  = data[test_idx]
    
    train_label_all = data_label[train_idx]
    test_label  = data_label[test_idx]
    
    input_dim  = data.shape[2]
    time_len   = data.shape[1]
    output_dim = data_label.shape[1]
    d_model    = d_model
    hidden_dim = hidden_dim
    nhead      = nhead
    hidden_dim_transformer = hidden_dim_transformer
    pos_drop    = pos_drop
    trans_drop = trans_drop
    fc_drop = fc_drop
    
    if torch.cuda.device_count() > 1:
        batch_size = 32 * torch.cuda.device_count()
    else:
        batch_size = 32
    
    test_dataset = MyDataset(test)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  num_workers=8,
                                                  persistent_workers=True)
    
    kf = KFold(n_splits=n_splits,shuffle=False)
    test_pred_all = []
    test_pred_all_corrected = []
    dt_now = datetime.datetime.now()

    for n_fold, (train_index, val_index) in enumerate(kf.split(train_all)):
        if multi:
            with open(f"output/age_predict_transformer_multi_smooth.txt", 'a') as f:
                f.write(dt_now.strftime('    %Y年%m月%d日 %H:%M:%S')+ f'\n    n_fold:{n_fold}\n')
        else:
            with open(f"output/age_predict_transformer_single_smooth.txt", 'a') as f:
                f.write(dt_now.strftime('    %Y年%m月%d日 %H:%M:%S')+ f'\n    n_fold:{n_fold}\n')
            
        save_epochs = 0
        train_loss_plot = []
        val_loss_plot = []
        
        train, train_label = train_all[train_index], train_label_all[train_index]
        val,   val_label   = train_all[val_index],   train_label_all[val_index]
        
        train_dataset    = MyDataset(train,train_label)
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=True,
                                                       num_workers=8,
                                                       persistent_workers=True)
    

        val_dataset      = MyDataset(val,val_label)
        val_dataloader   = torch.utils.data.DataLoader(val_dataset,
                                                       batch_size=batch_size,
                                                       num_workers=8,
                                                       persistent_workers=True)
        
        model = Model_multi(input_dim = input_dim,
                      d_model  = d_model,
                      hidden_dim = hidden_dim,
                      time_len = time_len,
                      nhead = nhead,
                      output_dim = output_dim,
                      hidden_dim_transformer = hidden_dim_transformer,
                      pos_drop = pos_drop,
                      trans_drop = trans_drop,
                      fc_drop = fc_drop
                     )
        
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)
        
        es = EarlyStopping(patience=200, verbose=1, mode='loss')
        best_val_loss = 1e5
        
        for epoch in range(epochs):
            train_loss = 0.
            val_loss = 0.

            for (x, t) in train_dataloader:
                x, t = x.to(device), t.to(device)
                if multi:
                    loss, preds_age, preds_sex = train_step_multi(x,t,model,optimizer)
                else:
                    loss, preds_age, preds_sex = train_step_single(x,t,model,optimizer)
                train_loss += loss.item()
            train_loss /= len(train_dataloader)
            
            for (x, t) in val_dataloader:
                x, t = x.to(device), t.to(device)
                if multi:
                    loss, preds_age, preds_sex = val_step_multi(x,t,model)
                else:
                    loss, preds_age, preds_sex = val_step_single(x,t,model)
                val_loss += loss.item()
            val_loss /= len(val_dataloader)
            
            if (epoch+1) % 200 == 0:
                dt_now = datetime.datetime.now()
                print(f'Kfold: {n_fold+1} ::: epoch: {epoch+1}, loss: {train_loss}, val loss: {val_loss}')
                
            #early stopping
            if epoch <=200:
                pass
            elif es(val_loss):
                break

            # modelの保存
            if epoch <=200:
                pass
            elif val_loss < best_val_loss:
                save_epochs = epoch+1
                best_val_loss = val_loss
                torch.save(model.module.state_dict(), model_path)
                
            train_loss_plot.append(train_loss)
            val_loss_plot.append(val_loss)
        
        model = Model_multi(input_dim = input_dim,
                      d_model =d_model,
                      hidden_dim = hidden_dim,
                      time_len = time_len,
                      nhead = nhead,
                      output_dim = output_dim,
                      hidden_dim_transformer = hidden_dim_transformer,
                      pos_drop = pos_drop,
                      trans_drop = trans_drop,
                      fc_drop = fc_drop
                     )
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        
        train_loss_plot = np.array(train_loss_plot)
        val_loss_plot   = np.array(val_loss_plot)
        x               = np.linspace(0, len(train_loss_plot), len(train_loss_plot))
        

        train_pred_age = []
        train_pred_sex = []
        train_label = []
        
        for (x, t) in train_dataloader:
            x, t = x.to(device), t.to(device)
            if multi:
                loss, preds_age, preds_sex = val_step_multi(x,t,model)
            else:
                loss, preds_age, preds_sex = val_step_single(x,t,model)
            train_pred_age.append(preds_age)
            train_pred_sex.append(preds_sex)
            train_label.append(t)
        
        val_pred_age = []
        val_pred_sex = []
        for (x, t) in val_dataloader:
            x, t = x.to(device), t.to(device)
            if multi:
                loss, preds_age, preds_sex = val_step_multi(x,t,model)
            else:
                loss, preds_age, preds_sex = val_step_single(x,t,model)
            val_pred_age.append(preds_age)
            val_pred_sex.append(preds_sex)
        
        train_pred_age = torch.cat(train_pred_age)
        train_pred_age = train_pred_age.to('cpu').detach().numpy().copy()
        train_pred_sex = torch.cat(train_pred_sex)
        train_pred_sex = train_pred_sex.to('cpu').detach().numpy().copy()
        
        train_label = torch.cat(train_label)
        train_label = train_label.to('cpu').detach().numpy().copy()
        
        val_pred_age = torch.cat(val_pred_age)
        val_pred_age = val_pred_age.to('cpu').detach().numpy().copy()
        val_pred_sex = torch.cat(val_pred_sex)
        val_pred_sex = val_pred_sex.to('cpu').detach().numpy().copy()

        lr = LinearRegression()
        lr.fit(val_label[:,0:1]*age_std+age_mean,val_pred_age[:,0]*age_std+age_mean)
                
        test_pred_age = []
        test_pred_sex = []
        
        for x in test_dataloader:
            x = x.to(device)
            if multi:
                preds_age, preds_sex = test_step_multi(x,model)
            else:
                preds_age, preds_sex = test_step_single(x,model)
            test_pred_age.append(preds_age)
            test_pred_sex.append(preds_sex)
        test_pred_age = torch.cat(test_pred_age)
        test_pred_sex = torch.cat(test_pred_sex)
        test_pred_age = test_pred_age.to('cpu').detach().numpy().copy()
        test_pred_sex = test_pred_sex.to('cpu').detach().numpy().copy()
        test_pred_age_corrected = (test_pred_age*age_std+age_mean - lr.intercept_)/ lr.coef_[0]
        test_pred_age_corrected = (test_pred_age_corrected-age_mean) / age_std
        test_pred     = np.concatenate([test_pred_age, test_pred_sex],1)
        test_pred_corrected  = np.concatenate([test_pred_age_corrected, test_pred_sex],1)
        if multi:
            if lr.coef_[0] > 0.1:
                test_pred_all.append(test_pred)
                test_pred_all_corrected.append(test_pred_corrected)
        else:
            test_pred_all.append(test_pred)
            test_pred_all_corrected.append(test_pred_corrected)

    return np.array(test_pred_all),np.array(test_pred_all_corrected),test_label