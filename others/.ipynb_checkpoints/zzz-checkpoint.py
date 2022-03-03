import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import os
print(f'os.cpu_count():{os.cpu_count()}')
print(f'len(os.sched_getaffinity(0)):{len(os.sched_getaffinity(0))}')

# Parameters and DataLoaders
input_size = 50000#500→50000
print(f'input_size:{input_size}\n')
output_size = 2
num_workers = 4
data_size = 800
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return self.len

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
    def forward(self, input):
        output = self.fc(input)
        return output
##################################################################
batch_size = 40
rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True,
                         num_workers=num_workers,persistent_workers=True
                                                 )
print(f'batch_size:{batch_size},num_workers:{num_workers} \n')
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.to(device)

start = time.time()

for i in range(100):
    for data in rand_loader:
        input = data.to(device)
        output = model(input)
    
elapsed_time = time.time() - start
print ("DataParallel : elapsed_time:{0}".format(elapsed_time) + "[sec]\n")

model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
model.to(device)

start = time.time()

for i in range(1000):
    for data in rand_loader:
        input = data.to(device)
        output = model(input)
    
elapsed_time = time.time() - start
print ("No DataParallel : elapsed_time:{0}".format(elapsed_time) + "[sec]\n")

##################################################################
batch_size = 160
rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True,
                         num_workers=num_workers,persistent_workers=True
                                                 )
print(f'batch_size:{batch_size},num_workers:{num_workers} \n')
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.to(device)

start = time.time()

for i in range(100):
    for data in rand_loader:
        input = data.to(device)
        output = model(input)
    
elapsed_time = time.time() - start
print ("DataParallel : elapsed_time:{0}".format(elapsed_time) + "[sec]\n")
    


model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
model.to(device)

start = time.time()

for i in range(1000):
    for data in rand_loader:
        input = data.to(device)
        output = model(input)
    
elapsed_time = time.time() - start
print ("No DataParallel : elapsed_time:{0}".format(elapsed_time) + "[sec]\n")

##################################################################
batch_size = 40
num_workers = 16
rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True,
                         num_workers=num_workers,persistent_workers=True
                                                 )
print(f'batch_size:{batch_size},num_workers:{num_workers} \n')
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.to(device)

start = time.time()

for i in range(100):
    for data in rand_loader:
        input = data.to(device)
        output = model(input)
    
elapsed_time = time.time() - start
print ("DataParallel : elapsed_time:{0}".format(elapsed_time) + "[sec]\n")

model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
model.to(device)

start = time.time()

for i in range(1000):
    for data in rand_loader:
        input = data.to(device)
        output = model(input)
    
elapsed_time = time.time() - start
print ("No DataParallel : elapsed_time:{0}".format(elapsed_time) + "[sec]\n")

##################################################################
batch_size = 160
num_workers = 16
rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True,
                         num_workers=num_workers,persistent_workers=True
                                                 )
print(f'batch_size:{batch_size},num_workers:{num_workers} \n')
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.to(device)

start = time.time()

for i in range(100):
    for data in rand_loader:
        input = data.to(device)
        output = model(input)
    
elapsed_time = time.time() - start
print ("DataParallel : elapsed_time:{0}".format(elapsed_time) + "[sec]\n")
    
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
model.to(device)

start = time.time()

for i in range(1000):
    for data in rand_loader:
        input = data.to(device)
        output = model(input)
    
elapsed_time = time.time() - start
print ("No DataParallel : elapsed_time:{0}".format(elapsed_time) + "[sec]\n")