# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 09:55:54 2024

@author: Chris
"""

import os
import pandas as pd


# Get dfs - but sample limited amount for 
def getData():

    dataset_folder = os.path.join(os.getcwd(), "## Data")
    
    # Get a list of all csv files - no. of csv files = no. of classes
    all_files = os.listdir(dataset_folder)
    csv_files = [file for file in all_files if file.endswith(".csv")]
    classes = [os.path.splitext(file)[0] for file in csv_files]
    
    dfs = []
    
    for csv_file in csv_files:
        
        df = pd.read_csv(os.path.join(dataset_folder, csv_file))
        df = df.fillna(0)
        dfs.append(df)
        
    return dfs, classes


dfs, classes = getData()



# List to hold sampled DataFrames
sampled_dfs = []

for df, label in zip(dfs, classes):
    
    n = 2000
        
    # Sampling n_benign rows randomly
    sampled_df = df.sample(n=n, random_state=42)  
    sampled_dfs.append(sampled_df)


result_df = pd.concat(sampled_dfs)

# Shuffle df
result_df_shuffled = result_df.sample(frac=1, random_state=97)

# Write to csv
result_df_shuffled.to_csv('sampled_data_v3.csv', index=False) 


