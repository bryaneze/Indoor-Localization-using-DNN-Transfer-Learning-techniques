## Installation guide 
- Jupyter notebooks can be directly run from Google Colab 
- Place datasets into a folder on Google Drive and mount it on Colab when running the jupyter notebook files to access it 

## Project description
- Exploring and implementing Indoor localization with Wi-Fi & BLE RSSI using Deep Neural Networks (DNN) and Transfer Learning (TL)


[National Gallery]
- Using collected data to experiment on DNN techniques to improve the accuracy of the indoor localization 

[SCALE]
- Experimenting DNN techniques on collected BLE RSSI values for indoor localization 

[UJIndoor dataset]
- Using UJIndoor dataset to test on pre-processing & DNN/TL techniques 

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
    └── README.md <-- this file
```

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
    - Latitude & Longitude for National Gallery and SCALE datasets were converted from EPSG4326 (default GPS) to EPSG3414 (Singapore meter based projected coordinates). 

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


### References used: 
- Udacity DNN course (Lessons 1 - 5)
- References in my FYP report (SCSE21-0138)
