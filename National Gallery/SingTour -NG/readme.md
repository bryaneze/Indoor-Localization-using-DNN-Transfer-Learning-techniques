# Instructions to run 

files: 

- `readme.md` <-- this file

- `ng_dnn.py` 

- `NG_MODEL.pt`

- `ng_dnn_notebook_REFERENCE.ipynb`



## `ng_dnn.py` 
- `ng_dnn.py` has 2 main functions: `load_model(model)` and `localisation(rssi_list, model)` 

- `load_model(model)` takes in 1 argument, where it should be the location of the model `NG_MODEL.pt` 

- `localisation(rssi_list, model)` takes 2 arguments, a list of 345 RSSI values and the model 


Check out `testCode()` function to see how the functions are used. A Jupyter notebook file is also included to see the outputs

### steps: 

1. load model by calling `load_model(model)` 

2. generate prediction by calling `localisation(rssi_list, model)` and the output would be `[[lon lat]]`



## `NG_MODEL.pt`
- Model for DNN 

## `ng_dnn_notebook.ipynb`
- For reference to see the outputs and such 