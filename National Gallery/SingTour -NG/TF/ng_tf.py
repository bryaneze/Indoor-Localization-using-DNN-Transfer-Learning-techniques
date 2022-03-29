import torch
import pandas as pd 
import torchvision
from torchvision import datasets, transforms
from torch import nn,optim
import torch.nn.functional as F
import numpy as np
import math


#Class for creating the model with the structure listed below
class Classifier(nn.Module):
    '''
    Returns
    ----------
    dnn_model : model 
        structure of the DNN model 
        
        input layer (345, 300) 
        dropout(.50) 
        hidden layer1 (300, 300)
        dropout(.50)
        hidden layer2 (300, 300)
        dropout(.50)
        output layer (300, 2)
    '''
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(345,300)
        self.fc2 = nn.Linear(300,300)
        self.fc3 = nn.Linear(300,300)
        self.fc4 = nn.Linear(300,2)

        
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self,x):
        x = x.view(x.shape[0],-1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)
        
        return x
    

    
def load_model(model_state): 
    '''
    Load trained model for localisation
    Parameters
    ----------
    model_state: String path of saved model state

    Returns model: load it into localisation()
    -------

    '''
    model = Classifier() 
    model.load_state_dict(torch.load(model_state))
    return model     


def scale_data(rss_df):
    if not('pandas' in str(type(rss_df))):
        print("ERROR: Please Input Data in pandas.DataFrame")
        return -1

    new_rss = rss_df.copy()
    new_rss[new_rss == 100] = -110
    new_rss[new_rss == -100] = -110

    new_rss /= 110
    new_rss += 1
    return new_rss.values

  
def rescale_xy(scale_lon_lat, mean, std):
    scale_lon_lat = np.array(scale_lon_lat)
    lon = np.transpose(np.array([scale_lon_lat[:, 0]]))
    # lon = output[0]
    lon_rescaled = lon * std[0] + mean[0]

    lat = np.transpose(np.array([scale_lon_lat[:, 1]]))
    # lat = output[1]
    lat_rescaled = lat * std[1] + mean[1]
    return np.array([lon_rescaled, lat_rescaled]).transpose().reshape(-1,2)





def localisation(rssi_list, model): 
    '''
    Parameters
    ----------
    rssi_list: list containing 345 RSSI values
    model: loaded trained model state

    Returns a list of predicted location i.e. [longitude, latitude] in EPSG3414
    -------

    '''
    #Very sketchy workabout to use the scale_data function
    #convert list into dataframe, call scale_data function, convert back to list 
    temp_df = pd.DataFrame(rssi_list)
    scaled_df = scale_data(temp_df)
    temp_list = scaled_df.tolist()
    rss = np.array(temp_list)

    #hardcoded mean and std 
    mean = [30032.69349262 ,30317.32534712]
    mean_np = np.array(mean)
    std = [22.18997454,36.97402449]
    std_np = np.array(std)

    rss_tensor = torch.from_numpy(rss).reshape(-1,345)
    with torch.no_grad():
        model.eval()
        test_input = rss_tensor.type(torch.FloatTensor)
        predict = model(test_input)
        final = rescale_xy(predict.data.numpy(), mean_np, std_np)
    return final 



#test codes on how to use the methods
def testCode(): 
    #Dataframe
    val_data = pd.read_csv('/content/drive/MyDrive/FYP/NG/Combined_collections_validation_EPSG3414_filtered.csv')

    #Add the 345 RSSI values into the list 
    rssi_list = val_data.iloc[0,:345].values.tolist() 

    #Load the model 
    model = load_model("/content/drive/MyDrive/FYP/NG/Model_Floor_2_BASE.pt")
    
    #model = load_model("/content/drive/MyDrive/FYP/NG/Model_Floor_1_TF.pt")  <------ Use this if testing on Level 1 
    #model = load_model("/content/drive/MyDrive/FYP/NG/Model_Floor_B1_TF.pt") <------ Use this if testing on Level B1
    #model = load_model("/content/drive/MyDrive/FYP/NG/NG_MODEL") <------------------ DNN model trained on ALL floors 

    #Run localisation function
    print(localisation(rssi_list, model))




