#https://note.com/yo16/n/n53259a32e12e
import sys
import datetime
import concurrent
from concurrent import futures

lis = list(sys.argv)
s = lis[1]

dt_now = datetime.datetime.now()
print(dt_now.strftime('%Y年%m月%d日 %H:%M:%S'))
print(f'start {s}')

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from pathlib import Path
import os

print(f'len(os.sched_getaffinity(0)):{len(os.sched_getaffinity(0))}')

#ファイル名からcsvファイルを取得＋subIDを付加
def file2df_ver2(data_files):
    df_list = []
    for t in data_files:
        df_temp = pd.read_csv(t,header = None)
        
        t_ = str(t)
        fd0000 = t_.find('0000')
        fdD    = t_.find('D',fd0000)
        fd_    = t_.find('_',fd0000)
        
        s = t_[fdD:fd_]
        df_temp.insert(0,'subID',s)
        
        df_list.append(df_temp)
    return pd.concat(df_list)

#ファイル名からcsvファイルを取得＋subIDを付加
def file2df(data_files):
    df_list = []
    for t in data_files:
        df_temp = pd.read_csv(t,header = 1)
        
        t_ = str(t)
        fd0000 = t_.find('0000')
        fdD    = t_.find('D',fd0000)
        fd_    = t_.find('_',fd0000)
        
        s = t_[fdD:fd_]
        df_temp.insert(0,'subID',s)
        
        df_list.append(df_temp)
    return pd.concat(df_list)

#columnsに付いている無駄なスペースを削除
def rem_space_columns(df):
    lis_new = []
    for i in df.columns:
        lis_new.append(i.replace(' ',''))
    df.set_axis(lis_new, axis='columns',inplace = True)
    return df

if s == 'aal' or s == 'ho':
    data_path = Path('../01_data/roi_timeseries')
    data_files = (data_path).rglob(f'_selector_CSF-2mm-M_aC-CSF+WM-2mm-DPC5_M-SDB_P-2_BP-B0.01-T0.1/\
_mask_{s}_mask_pad_mask_file_..resources..{s}_mask_pad.nii.gz/roi_stats.csv')
    df = file2df(data_files)
    df = rem_space_columns(df)
    df.reset_index(inplace=True)
else:
    data_path = Path('../01_data/roi_timeseries_others')
    data_files = (data_path).rglob(f'{s}/*.csv')
    df = file2df_ver2(data_files)
    df.reset_index(inplace=True)    

df_subjects_data= pd.read_excel('../01_data/shimane_basicInfo/RevisedNew_Shimane3TsubjectsData.xlsx',
                                engine='openpyxl')
df_subjects_data.drop('Unnamed: 102',axis=1,inplace=True)

def standardization(df):
    for subID in df.subID.unique():
        df_ = df[df['subID'] == subID]
        df_ = df_.drop(['index','subID'],axis=1)
        data = np.array(df_)
        data -= np.mean(data)
        data /= np.std(data)
        roi_name = [i for i in list(df.columns).copy() if i != 'index' and i != 'subID']
        df.loc[df['subID'] == subID,roi_name] = data
    return df

df = standardization(df)
df.rename(columns={'index': 'time'}, inplace=True)

def dynamic_FC_each(lis_col,lis,c,num_take,window_size,subID):
    dt_now = datetime.datetime.now()
    lis_temp = lis[c*num_take:c*num_take+num_take]#ある被験者についてのtimeseries情報
    lis_dynamic_each = [[]] * (len(lis_temp)-window_size+1)
    for i in range(len(lis_temp)-window_size+1):
        lis_dynamic_temp = lis_temp[i:i+window_size]
        lis_df = []#計算後DataFrameとなるlist
        lis_df.append(f'{i}-{i+window_size-1}')
        lis_df.append(f'{subID}')
        for j in range(len(lis_col)):
            lis_dynamic_temp1 = [l[j] for l in lis_dynamic_temp]
            for k in range(j+1,len(lis_col)):
                lis_dynamic_temp2 = [l[k] for l in lis_dynamic_temp]
                corr,_ = pearsonr(lis_dynamic_temp1,lis_dynamic_temp2)
                if np.isnan(corr):
                    lis_df.append(0)
                else:
                    lis_df.append(corr)
        lis_dynamic_each[i] = lis_df
    return [c,lis_dynamic_each]

def dynamic_FC_fast(df,window_size = 18,extra = 2):
    '''
    window_size：TR=2.5より、45秒間
    extra：dfのcolumnsの最初のindex,subIDを無視
    dynamic_FC：被験者あたり２４分かかる(ROIの数に依る)
    dynamic_FC_fast：被験者あたり２分以下で終わる(ROIの数に依る)
    '''
    lis_col = list(df.columns)[extra:]#ROI名
    lis_col_new = list(df.columns)[:extra]#dynamic_FCのcolumns
    
    for m in range(len(lis_col)):
        for n in range(m+1,len(lis_col)):
            lis_col_new.append(str(lis_col[m]) + '_' + str(lis_col[n]))
    
    lis = df.iloc[:,2:].values.tolist()#全データをlistに変換
    num_take = int(len(lis)/len(df.subID.unique()))#一人あたり撮像回数
    sub_list = list(df.subID.unique())
    
    with futures.ProcessPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as executor:
        lis_dynamic = [[]]*len(sub_list)*(num_take-window_size+1)
        rets = []
        for c, subID in enumerate(sub_list):
            rets.append(executor.submit(fn=dynamic_FC_each,
                            lis_col=lis_col,
                            lis=lis,
                            c=c,
                            num_take=num_take,
                            window_size=window_size,
                            subID=subID
                           ))
    rets2 = []
    for ret in concurrent.futures.as_completed(rets):
        rets2.append(ret.result())
    for ret in rets2:
        c = ret[0]
        lis_dynamic[c*(num_take-window_size+1):c*(num_take-window_size+1)+(num_take-window_size+1)] = ret[1]
    df_dynamic = pd.DataFrame(lis_dynamic,columns=lis_col_new)
    return df_dynamic

df_dynamic = dynamic_FC_fast(df)
df_dynamic.rename(columns={'index': 'time'}, inplace=True)
df_dynamic.reset_index(inplace=True)
df_dynamic.drop('index',axis=1,inplace=True)

def subjects_once(df = df, df_sub = df_subjects_data):
    sub_lis = list(df[['subID']].drop_duplicates()['subID'])
    for i in range(df_sub.shape[0]):
        subID = df_sub['subID'][i][df_sub['subID'][i].find('D'):]
        if subID in sub_lis:
            df_sub.loc[i,'subID'] = df_sub['subID'][i][df_sub['subID'][i].find('D'):]
    return df_sub[[False if i[0]=='2' else True for i in df_sub['subID']]]#入ってないやつは2から始まる

def subjects_not_in(df,df_sub=df_subjects_data):
    lis_not_in = list(set(list(df[['subID']].drop_duplicates()['subID'])) - set(df_sub.subID))
    return df[[False if i in lis_not_in else True for i in df['subID']]]

df_subjects_data = df_subjects_data.rename(columns={'NewID': 'subID'})
df_subjects_data = subjects_once(df = df, df_sub = df_subjects_data)
df_subjects_data = df_subjects_data.sort_values('subID').reset_index(drop=True)

df = subjects_not_in(df=df,df_sub=df_subjects_data)
df = df.sort_values(['subID','time']).reset_index(drop=True)

time_dfc_lis = [int(i.split('-')[0]) for i in df_dynamic.time]
df_dynamic.time = time_dfc_lis
df_dynamic = subjects_not_in(df=df_dynamic,df_sub=df_subjects_data)
df_dynamic = df_dynamic.sort_values(['subID','time']).reset_index(drop=True)

#順番をごちゃまぜにする
def sort_df(df):
    np.random.seed(0)
    sort = np.array(df.subID.drop_duplicates())
    sort = np.random.choice(sort, len(sort), replace=False)
    sort = list(sort)
    
    if 'time' in df.columns:
        df['sort'] = df['subID'].apply(lambda x: sort.index(x) if x in sort else -1)
        df = df.sort_values(['sort','time']).reset_index(drop=True).drop('sort', axis=1)
    else:
        df['sort'] = df['subID'].apply(lambda x: sort.index(x) if x in sort else -1)
        df = df.sort_values(['sort']).reset_index(drop=True).drop('sort', axis=1)
    return df

df = sort_df(df)
df_subjects_data = sort_df(df_subjects_data)
df_dynamic = sort_df(df_dynamic)

def transform_float(df):
    keys = list(df.columns)
    keys.remove('time')
    keys.remove('subID')
    values = ['float16']*len(keys)
    d = dict(zip(keys, values))
    df = df.astype(d)
    return df

df_dynamic = transform_float(df_dynamic)

df = df[df['subID'] != df_subjects_data[df_subjects_data['Age'] == 20]['subID'].iloc[0]]
df_dynamic = df_dynamic[df_dynamic['subID'] != df_subjects_data[df_subjects_data['Age'] == 20]['subID'].iloc[0]]
df_subjects_data = df_subjects_data[df_subjects_data['subID'] != df_subjects_data[df_subjects_data['Age'] == 20]['subID'].iloc[0]]

df.to_csv(f'/gs/hs0/tga-akamalab/shimane_data/roi_timeseries/timeseries_{s}.csv', index=False)
df_dynamic.to_csv(f'/gs/hs0/tga-akamalab/shimane_data/dynamic_FC/dynamic_{s}.csv', index=False)

dt_now = datetime.datetime.now()
print(dt_now.strftime('%Y年%m月%d日 %H:%M:%S'))
print(f'finish {s}')