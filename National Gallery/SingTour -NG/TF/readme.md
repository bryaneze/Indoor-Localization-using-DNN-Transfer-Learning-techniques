# Instructions to run 

files: 

- `readme.md` <-- this file

- `ng_tf.py` 

- `Model_Floor_2_BASE.pt`

- `Model_Floor_1_TF.pt`

- `Model_Floor_B1_TF.pt`




## `ng_tf.py` 
- `ng_tf.py` has 2 main functions: `load_model(model)` and `localisation(rssi_list, model)` 

- `load_model(model)` takes in 1 argument, where it should be the location of the model 

- `localisation(rssi_list, model)` takes 2 arguments, a list of 345 RSSI values and the model 


Check out `testCode()` function to see how the functions are used.

### steps: 

1. load model by calling `load_model(model)` 

2. generate prediction by calling `localisation(rssi_list, model)` and the output would be `[[lon lat]]`



## `Model_Floor_2_BASE.pt` 
- Model for DNN Floor 2 base model 

## `Model_Floor_1_TF.pt`
- Model for TF Floor 1 model 

## `Model_Floor_B1_TF.pt`
- Model for TF Floor B1 model 

