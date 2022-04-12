import torch
import pandas as pd 
import torchvision
from torchvision import datasets, transforms
from torch import nn,optim
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
data = pd.read_csv('/path/to/csv')

label = data[["lon_3414","lat_3414"]]
#print(label)

data = data.drop(columns=['type','timestamp','latitude','longitude','lon_3414','lat_3414','floor_id'])
#print(data)

#convert df to tensor object
data = torch.tensor(data.values)

label_t = torch.tensor(label.values)

#flatten 
data_t = torch.flatten(data,1)

label_t = torch.flatten(label_t,1)
print(label_t.shape)
print(data_t.shape)


train_data = [] 
for i in range(len(data)):
    train_data.append([data_t[i],label_t[i]])




data_val = pd.read_csv('path/to/csv')
label_val=data_val[["lon_3414","lat_3414"]]
data_val=data_val.drop(columns=['type','timestamp','latitude','longitude','lon_3414','lat_3414','floor_id'])
#convert df to tensor object
data_val = torch.tensor(data_val.values)

label_val_t = torch.tensor(label_val.values)

#flatten 
data_val_t = torch.flatten(data_val,1)

label_val_t = torch.flatten(label_val_t,1)


val_data = [] 
for i in range(len(data_val)):
    val_data.append([data_t[i],label_t[i]])



	    
#put both train and validation sets into loader 
trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=32)
testloader = torch.utils.data.DataLoader(val_data,shuffle=True, batch_size=32)

print("Finish dataset part")
