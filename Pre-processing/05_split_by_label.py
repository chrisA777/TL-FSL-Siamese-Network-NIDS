import os
import pandas as pd
import random

# Days of the week
days = ['Tuesday', 'Wednesday']

# Num of files for each day (counting from 0)
n_files = [15, 19, 13, 14]

# Iterate through each of the days
for i in range(len(days)):
    
    # Folder where data is found
    folder_name = f'{days[i]}-WorkingHours'
    
    # Create output folder path
    output_folder_path = os.path.join(folder_name, 'Segregated by Label')
    os.makedirs(output_folder_path, exist_ok=True) 
    
    # Initialize an empty df to store concatenated data for the current day
    concatenated_data = pd.DataFrame()
    
    # Iterate through each file for the current day
    for file_num in range(0, n_files[i] + 1):
          
        # Define folder and file name
        file_name = f'{folder_name}_{file_num}_flows_v2.csv'
        file_path = os.path.join(os.getcwd(), folder_name, file_name)
        
        # Read the csv file
        df = pd.read_csv(file_path)
        
        # Etract benign data - treat seperately because of how much there is
        benign_df = df[df['categorical_label'] == 'benign']
        
        # If there is benign data, write this to seperate csv
        if not benign_df.empty:
            
            output_file_name = f'{file_name.replace(".csv", "_benign.csv")}'
            output_file_path = os.path.join(output_folder_path, output_file_name)
            
            benign_df.to_csv(output_file_path, index=False)
        
        # Exclude rows with benign label
        df = df[df['categorical_label'] != 'benign']
        
        if not df.empty:
            concatenated_data = pd.concat([concatenated_data, df], ignore_index=True)
    
    # Create output folder path
    output_folder_path = os.path.join(folder_name, 'Concatenated by Label')
    os.makedirs(output_folder_path, exist_ok=True)  # Ensure the output folder exists
    
    drop_columns = ['flow_idx', 'flow_id', 'packet_num', 'timestamp', 'datetime',
    'version', 'header_len', 'src_address', 'dst_address', 'tot_len', 'ttl',
    'tos', 'id', 'proto', 'ip_chksum', 'frag_offset', 'df_flag', 'mf_flag',
    'sport', 'dport', 'offset', 'window', 't_chksum', 'urp', 'urg_flag',
    'fin_flag', 'ack_flag', 'syn_flag', 'rst_flag', 'psh_flag', 'ece_flag',
    'cwr_flag', 'ns_flag', 'ulen']
    
    # Save the concatenated data to a file for each label
    grouped = concatenated_data.groupby('categorical_label')
    for label, group in grouped:
        
        group = group.drop(columns = drop_columns, errors = 'ignore')
        
        # Define the output file name
        output_file_name = f'{label}.csv'
        output_file_path = os.path.join(output_folder_path, output_file_name)
        
        group = group.drop_duplicates()
        
        # Save the group to the output file
        group.to_csv(output_file_path, index=False)
