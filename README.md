
## Project description
Exploring and implementing Indoor localization with Wi-Fi & BLE RSSI using Deep Neural Networks (DNN) and Transfer Learning (TL). 3 locations were considered for this project which are the subdirectories in the root directory. For National Gallery & SCALE, the dataset was collected using the data collection app and preprocessed. 

DNN/TF models were trained on Google Colab using datasets and Jupyter Notebook codes in the subdirectories. Packages used are listed in the section below, and in `requirements.txt`. 



## Installation/set-up
- Jupyter notebooks can be directly run from Google Colab 
- Place datasets into a folder on Google Drive and mount it on Colab when running the jupyter notebook files to access it 

### Packages used: (shown in requirements.txt)
- torch 1.10.0 
- pandas 1.3.5 
- torchvision 0.11.1 
- torch 1.10.1 
- numpy 1.21.5 
- pyproj 3.3.0 






##  Project directory structure 
```
    .
    ├── National Gallery 
    │   ├── Datasets   <- csv datasets collected & used 
    |   ├── DNN Implementation  <- implementation codes for DNN
    |   └── SingTour -NG  <- Codes for integration with SingTour app 
    |           ├── DNN <- DNN model for SingTour   
    |           └── TF <- Transfer Learning model for SingTour
    |
    ├── SCALE 
    |   ├── Datasets <- csv datasets collected & used 
    |   ├── Reference images <- images for beacon taggings 
    |   └── DNN Implementation <- implementation codes 
    |
    ├── UJI   
    |   ├── Datasets <- csv datasets used 
    |   ├── DNN Implementations <- implementation codes for DNN
    |   └── Transfer learning implementations <- implementation codes for TF
    |
    ├── requirements.txt <-- packages and versions 
    └── README.md <-- this file
```

[National Gallery](https://github.com/NTU-SCALE-Lost-And-Found/LnF_FYP2122S1_Bryan-Eng-Ze-En/tree/main/National%20Gallery)
- Using collected data to experiment on DNN techniques to improve the accuracy of the indoor localization 

[SCALE](https://github.com/NTU-SCALE-Lost-And-Found/LnF_FYP2122S1_Bryan-Eng-Ze-En/tree/main/SCALE)
- Experimenting DNN techniques on collected BLE RSSI values for indoor localization 

[UJIndoor dataset](https://github.com/NTU-SCALE-Lost-And-Found/LnF_FYP2122S1_Bryan-Eng-Ze-En/tree/main/UJI)
- Using UJIndoor dataset to test on pre-processing & DNN/TL techniques 



## Techniques explored/experimented: 
### Preprocessing
After doing data collection at National Gallery, we have over 800 unique APs. Therefore there is a need to do some sort of data processing/feature selection to reduce the number of unwanted APs so that the DNN model would be able to perform better. Preprocessing techniques experimented are as follows: 

- Recursive Feature Elimination (RFE)
    - implementation of RFE on National Gallery dataset in `National Gallery/Datasets/RFE/` 
    - using RandomForestRegressor estimator 
    - select top 400 features 
- Threshold filtering 
    - implementation of Threshold filtering on National Gallery dataset in `National Gallery/Datasets/filter/` 
    - remove APs that has no values more than threshold (-70dBM) across all runs/routes collected 

- Conversion of GPS Coordinates 
    - Latitude & Longitude for National Gallery and SCALE datasets were converted from EPSG4326 (default GPS) to EPSG3414 (Singapore meter based projected coordinates). Codes for conversion of GPS can be found in `SCALE/DNN Implementation/EPSGconveter.py`

### DNN 
- DNN implementation codes are defined in `National Gallery/DNN Implementation/` & `UJI/DNN Implementation`
- Tested with both unprocessed and processed datasets 
- DNN model tested: 
    -  Input layer with N numbers of AP RSSI values 
    -  3 hidden layers `⌊N⌋ (to the nearest 100) - 100` with ReLU activation function 
    -  output layer with 2 neurons that represents longitude and latitude 

![image](https://user-images.githubusercontent.com/26837821/144977985-59b2de77-9945-43d8-86fe-1527ece797fa.png)


### Transfer learning 
- Transfer learning implementation codes are in `National Gallery/SingTour -NG/TF` & `UJI/Transfer learning implementations`
- Transfer learning techniques used: 
    - Tuning 
    - Feature selection 

## Report link: 
- Eng,  B. Z. E. (2022). Indoor localization and navigation via Wi-Fi & bluetooth fingerprinting. Final Year Project (FYP), Nanyang Technological University, Singapore. https://hdl.handle.net/10356/156699


## References used: 
- [Udacity PyTorch course](https://www.udacity.com/course/deep-learning-pytorch--ud188) 


