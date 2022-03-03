import torch
import torch.nn as nn
import datetime

dt_now = datetime.datetime.now()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import sys
lis = list(sys.argv)

if len(lis) == 1:
    with open("output/age_predict.txt", 'r+') as f:
        f.truncate(0)
    with open("output/age_predict.txt", 'a') as f:
        f.write(dt_now.strftime('%Y年%m月%d日 %H:%M:%S') + '\n' + f'device：{device}' + '\n' + f'device_count：{torch.cuda.device_count()}' + '\n' )
    print(f'device：{device}')
    print(f'device_count：{torch.cuda.device_count()}')
elif len(lis) == 2:
    s = lis[1]
    print(s)
    with open("output/age_predict.txt", 'a') as f:
        f.write(dt_now.strftime('%Y年%m月%d日 %H:%M:%S') + '\n' + s + '\n')