from calendar import month_abbr
from csv import reader
from os import listdir, mkdir
from os.path import join, isfile, isdir
from pathlib import Path
from random import choice, shuffle
from shutil import make_archive, rmtree
from zipfile import ZipFile
import math
import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas import DataFrame as df
from pandas.core.algorithms import unique 

def removing():
    df = pd.read_csv("./SG_MAP_COORDS/training.csv")

    index_list = [] 
    counter = 0
    for i in range(847):
        # s = df[f'AP{i}']
        if (df[f'WAP{i}'] > -70).any():
            continue
        else:
         index_list.append(i)
         df.drop(f'WAP{i}', axis=1, inplace=True)


    df.to_csv("filter-train.csv",index=False)
    return index_list


def remove_val(index_list):
    val_df = pd.read_csv("./SG_MAP_COORDS/validation.csv")
    val_df.drop(val_df.columns[index_list], axis = 1, inplace = True)
    val_df.to_csv("filter-test.csv", index=False)

remove_val(removing())



# processed_test = pd.read_csv("./processed_test.csv")
# unique_AP = pd.read_csv("unique_AP_combined.csv")
# processed_test.drop(columns=processed_test.columns[0],axis =1, inplace=True)


# unique_AP.drop(columns=unique_AP.columns[0],axis =1,inplace=True)
#print(unique_AP)



# #remove AP at the front so can use .isin method 

# #for processed dataframe: 
# processed_test.columns = processed_test.columns.str.strip('AP')
 

# #for unique_AP dataframe: 
# unique_AP['ap'] = unique_AP['ap'].str[2:]

# #print(unique_AP['ap'])
# first_row = (list(processed_test))
# del first_row[-1]
# del first_row[-1]
# del first_row[-1]
# del first_row[-1]
# #print(len(first_row))

# unique_AP = unique_AP[unique_AP['ap'].isin(first_row)]
# print((unique_AP['ap']))
# unique_AP['ap'] = ['AP{}'.format(i) for i in range(unique_AP.shape[0])]
# print((unique_AP['ap']))


# #print(processed_test.columns)
# processed_test.columns = ['AP{}'.format(i) for i in range(processed_test.shape[1])]
# # print(processed_test['AP421'])
# # print(processed_test['AP420'])
# # print(processed_test['AP419'])
# # print(processed_test['AP418'])


# processed_test.rename(columns={'AP421':'timestamp'}, inplace=True)
# processed_test.rename(columns={'AP420':'type'}, inplace=True)
# processed_test.rename(columns={'AP419':'longitude'}, inplace=True)
# processed_test.rename(columns={'AP418':'latitude'}, inplace=True)





# processed_test.to_csv("filtered_processed_test.csv")
#unique_AP.to_csv("filtered_unique_AP.csv")
