# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 18:08:09 2023

@author: Chris
"""
import subprocess
import os
import sys

import pandas as pd


def reorder_columns(df, first_cols):
    return df[first_cols + [col for col in df.columns if col not in first_cols]]

# Function to label each packet based on flow
def identify_flows(file_name):
    
    # Read in a given csv 
    csv_path = os.path.join(os.getcwd(), file_name, file_name)
    df = pd.read_csv(csv_path + '.csv')
    print('Read csv')
    
    # Create a new column 'flow_id' to store the flow identifier
    df['flow_id'] = None
    df['flow_idx'] = None

    # Initialize a dictionary to track flow identifiers
    flow_dict = {}
    
    # Iterate through all packets
    for i in range (len(df['packet_num'])):
        
        # Process check
        if (i%100000 == 0 ):
            print(i)
        
        # Extract source, destination, and transport protocol
        src_ip = df['src_address'][i]
        dst_ip = df['dst_address'][i]
        src_port = df['sport'][i]
        dst_port = df['dport'][i]
        protocol = df['proto'][i]
        
        # 'Labelling' packets by flow 
        key1 = (src_ip, dst_ip, src_port, dst_port, protocol) 
        key2 = (dst_ip, src_ip, dst_port, src_port, protocol)                   # account for both flow directions
        
        key1 = "_".join(map(str, key1))                                         # Need to use a string as dict key
        key2 = "_".join(map(str, key2))
        
        # Check if key already defined
        if (key1 in flow_dict):                                                 # Check for first direction
            flow_id = flow_dict[key1]                                         
        elif (key2 in flow_dict):                                               # And also check for opposite direction
             flow_id = flow_dict[key2]   
        else:                                                                   # If neither exist
            
            flow_id = len(flow_dict)                                            # Create new one
            flow_dict[key1] = flow_id                                           # And add this to our dict
            
        
        # Assign the flow_id to the current packet
        df.at[i, 'flow_id'] = key1
        df.at[i, 'flow_idx'] = flow_id
    
    
    # Reorder column so that flow id is at the start 
    print('Reordering df...')
    column_order = ['flow_id', 'flow_idx']
    df = reorder_columns(df, column_order)
    
    # Sorting based on flow id
    print('Soring df...')
    df.sort_values(by = 'flow_idx', inplace = True)
    
    # Write the dict to csv
    print (' Writing to csv... ')
    
    
    csv_path = os.path.join(folder_path, file_name + '_sorted.csv')
    df.to_csv(csv_path, index = False)
    
   # print('Complete: Identified packet flows')
    return df


    
weekday = 'Thursday'


# Define folder name and path
file_name = weekday + '-WorkingHours'  
folder_path = os.path.join(os.getcwd(), file_name) 

a= identify_flows(file_name)

file = "split_parsed_csv.py"

with open (file, 'r') as file1:
    code = file1.read()
    exec(code)





