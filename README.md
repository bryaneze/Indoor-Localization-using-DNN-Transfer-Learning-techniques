# LnF_FYP2122S1_Bryan_Eng


## Project description
- Exploring and implementing Indoor localization using DNN techniques 


[National Gallery]
- Using collected data to experiment on DNN techniques to improve the accuracy of the indoor localization 

[SCALE]
- Experimenting DNN techniques on collected BLE RSSI values for indoor localization 

[UJIndoor dataset]
- Using UJIndoor dataset to test on pre-processing & DNN techniques 

##  Project directory structure 
```
    .
    ├── National Gallery 
    │   ├── Datasets   <- csv datasets collected & used 
    |   ├── DNN Implementation  <- implementation codes 
    |
    ├── SCALE 
    |   ├── Datasets <- csv datasets collected & used 
    |   ├── Reference images <- images for beacon taggings 
    |   ├── DNN Implementation <- implementation codes 
    |
    ├── UJI   
    |   ├── Datasets <- csv datasets used 
    |   ├── DNN Implementations <- implementation codes 
    └── README.md
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

### DNN (WIP)
- DNN implementation codes are defined in `National Gallery/DNN Implementation/` & `UJI/DNN Implementation`
- Tested with both unprocessed and processed datasets 
- DNN model tested: 
-  Input layer with N numbers of AP RSSI values 
-  3 hidden layers `⌊N⌋ (to the nearest 100) - 100` with ReLU activation function 
-  output layer with 2 neurons that represents longitude and latitude 

![image](https://user-images.githubusercontent.com/26837821/144977985-59b2de77-9945-43d8-86fe-1527ece797fa.png)


## TODO: 
- Transfer Learning implementation
- Add results for different implementations 
- `Setup process` & `Flow of function` sections 
