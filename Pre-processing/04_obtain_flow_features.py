# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 09:53:27 2023

@author: Chris
"""

import os
import pandas as pd

# Function used to position first_cols at front of df
# (not my code)
def reorder_columns(df, first_cols):
    return df[first_cols + [col for col in df.columns if col not in first_cols]]

def obtain_flow_features(file_path):
    
    i = 0 
    
    # Read the CSV file
    
    df = pd.read_csv(file_path + '.csv')
    df1 = df
    
    
    # Group data based on flow_idx and calculate uni flow statistics
    grouped_idx = df.groupby('flow_idx')
    bi_flow_stats = []

    # Calculate bi flow features
    for flow_idx, group_data_idx in grouped_idx:
        
        i+=1
        
        if (i%200 == 0):
            print(i)

        
        grouped_id = group_data_idx.groupby('flow_id')
        
        
        flow_ids = group_data_idx['flow_id'].unique()
        
        fwd_id = flow_ids[0]
        
        fwd = group_data_idx['flow_id'] == fwd_id
        bwd = ~fwd
        
        check_bwd = not all(fwd)
        
        # Comment & Uncomment features as required
        flow_stat = {
            
            'flow_idx': flow_idx,
            #'num_packets': len(group_data_idx),
            #'duration': group_data_idx['timestamp'].max() - group_data_idx['timestamp'].min(),
            #'total_bytes': group_data_idx['tot_len'].sum(),
            #'avg_pkt_length': group_data_idx['tot_len'].mean(),
            #'max_pkt_length': group_data_idx['tot_len'].max(),
            #'min_pkt_length': group_data_idx['tot_len'].min(),
            #'avg_inter_time': group_data_idx['timestamp'].diff().abs().dropna().mean(),
            #'std_inter_time': group_data_idx['timestamp'].diff().abs().dropna().std(),
            #'num_urg': group_data_idx['urg_flag'].sum(),
            #'num_ack': group_data_idx['ack_flag'].sum(),
            #'num_syn': group_data_idx['syn_flag'].sum(),
            #'num_rst': group_data_idx['rst_flag'].sum(),
            #'num_psh': group_data_idx['psh_flag'].sum(),
            
            'fwd_num_packets': len(group_data_idx[fwd]),
            'bwd_num_packets': len(group_data_idx[bwd]) if check_bwd else 0,
            
            'fwd_total_bytes': group_data_idx['tot_len'][fwd].sum(),
            'bwd_total_bytes': group_data_idx['tot_len'][bwd].sum() if check_bwd else 0,
            
            'fwd_avg_pkt_length': group_data_idx['tot_len'][fwd].mean(),
            'bwd_avg_pkt_length': group_data_idx['tot_len'][bwd].mean() if check_bwd else 0,
            
            'fwd_std_pkt_length': group_data_idx['tot_len'][fwd].std(),
            'bwd_std_pkt_length': group_data_idx['tot_len'][bwd].std() if check_bwd else 0,
            
            'fwd_max_pkt_length': group_data_idx['tot_len'][fwd].max(),
            'bwd_max_pkt_length': group_data_idx['tot_len'][bwd].max() if check_bwd else 0,
            
            'fwd_min_pkt_length': group_data_idx['tot_len'][fwd].min(),
            'bwd_min_pkt_length': group_data_idx['tot_len'][bwd].min() if check_bwd else 0,
            
            'fwd_avg_inter_time': group_data_idx['timestamp'][fwd].diff().abs().dropna().mean(),
            'bwd_avg_inter_time': group_data_idx['timestamp'][bwd].diff().abs().dropna().mean()
                                                            if check_bwd else 0,
            
            'fwd_std_inter_time': group_data_idx['timestamp'][fwd].diff().abs().dropna().std(),
            'bwd_std_inter_time': group_data_idx['timestamp'][bwd].diff().abs().dropna().std()
                                                            if check_bwd else 0,
                                                            
            'fwd_min_inter_time': group_data_idx['timestamp'][fwd].diff().abs().dropna().min(),
            'bwd_min_inter_time': group_data_idx['timestamp'][bwd].diff().abs().dropna().min() 
                                                            if check_bwd else 0,
            'fwd_max_inter_time': group_data_idx['timestamp'][fwd].diff().abs().dropna().max(),
            'bwd_max_inter_time': group_data_idx['timestamp'][bwd].diff().abs().dropna().max() 
                                                            if check_bwd else 0,   
            
            'fwd_avg_offset': group_data_idx['offset'][fwd].mean(),
            'bwd_avg_offset': group_data_idx['offset'][bwd].mean() if check_bwd else 0,
                                                                     
            #'fwd_num_urg': group_data_idx['urg_flag'][fwd].sum(),
            #'bwd_num_urg': group_data_idx['urg_flag'][bwd].sum() if check_bwd else 0,
            
            #'fwd_num_ack': group_data_idx['ack_flag'][fwd].sum(),
            #'bwd_num_ack': group_data_idx['ack_flag'][bwd].sum() if check_bwd else 0,
            
            #'fwd_num_syn': group_data_idx['syn_flag'][fwd].sum(),
            #'bwd_num_syn': group_data_idx['syn_flag'][bwd].sum() if check_bwd else 0,
            
            'fwd_num_rst': group_data_idx['rst_flag'][fwd].sum(),
            'bwd_num_rst': group_data_idx['rst_flag'][bwd].sum() if check_bwd else 0,
                
            'fwd_num_psh': group_data_idx['psh_flag'][fwd].sum(),
            'bwd_num_psh': group_data_idx['psh_flag'][bwd].sum() if check_bwd else 0,
        }
        
        bi_flow_stats.append(flow_stat)
    
    flow_statistics_df = pd.DataFrame(bi_flow_stats)

    # Merge dataframes to obtain flow and packet features
    df = df.merge(flow_statistics_df, on='flow_idx', how='left')

    # Reorganize column order
    column_order = flow_statistics_df.columns.tolist()
    df = reorder_columns(df, column_order)
    
    # Reorder specific columns
    column_order = ['flow_idx', 'flow_id']
    df= reorder_columns(df,column_order)
    
    # Write the result back to a CSV file
    df.to_csv(file_path + '_flows_v2.csv', index=False)
    
    print('Complete: Calculated flow level features')
    
    return df1, grouped_id, grouped_idx

    #print(f'Complete: new file available: {new_file_path}')
    
if __name__ == "__main__":
    
    weekday = 'Friday'
    n_files = 15
    
    folder_name = weekday +'-WorkingHours'
    
    for i in range(n_files):
        file_name = folder_name + f'_{i}'
        
        file_path = os.path.join(os.getcwd(), folder_name, file_name)
        
        df, a, b = obtain_flow_features(file_path)
    

    