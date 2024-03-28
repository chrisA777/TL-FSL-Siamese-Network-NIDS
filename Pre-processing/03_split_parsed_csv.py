import os
import pandas as pd

print('test')
# Define weekday
weekday = 'Friday'

# Define folder name and path to large .csv file
file_name = weekday + '-WorkingHours'  
folder_path = os.path.join(os.getcwd(), file_name)
csv_path = os.path.join(folder_path, file_name + '_sorted.csv')

# Open the CSV file for reading and count no. of rows
with open(csv_path, 'r') as f:
    # Get the total number of rows in the CSV file
    total_rows = sum(1 for line in f)
    

# Define size of chunks at which we will read csv (save time + avoid memory errors)
chunk_size = 2000000

# Calculate number of chunks and remainder
n_chunks, remainder = divmod(total_rows, chunk_size)

# If we have a remainder - need another chunk
if (remainder != 0):
    n_chunks+=1
    
    
# Initialize variables
flow_idx = None                                                                 # Variable to store current flow idx
current_file_index = 0                                                          # Idx of file we are writing to
max_packets_per_file = 500000                                                   # Max no of packets we want in a file
packets_written = 0                                                             # Count of how many packets we have written

# Defining path to output .csv files
output_path = f'{file_name}_{current_file_index}.csv'
output_path = os.path.join(folder_path, output_path)


# Create an empty list to store the packets for the current file
current_file_packets = []
    
for i in range(n_chunks):
                              
    # Defining range of rows to skip when reading csv in chunks
    if (i==0):
        start_row = 0                                                           # Dont skip any on first iteration 
        end_row = 0
    elif (i == n_chunks-1):                                                     # End row and chunk size differ for final iteration
        chunk_size = remainder
        end_row = total_rows - chunk_size
    else:
        start_row = 1
        end_row = i* chunk_size
        
    
    
    skip_range = range(start_row, end_row)
    
    df= pd.read_csv(csv_path, skiprows=skip_range, 
                    nrows=chunk_size)
     
    # Iterate through the DataFrame and split into separate CSV files
    for index, row in df.iterrows():

        if (packets_written % 10000 == 0):
           print(packets_written)
           
        if packets_written >= max_packets_per_file:
            # Find the last row with the same flow_idx
    
            if df['flow_idx'][index] != flow_idx:
                print(packets_written)
                
                # Write the current_file_packets to a CSV file
                df_to_write = pd.DataFrame(current_file_packets)
                df_to_write.to_csv(output_path, index=False)
    
                # Increment the file index and reset variables
                current_file_index += 1
                packets_written = 0
                current_file_packets = []
                output_path = f'{file_name}_{current_file_index}.csv'
                output_path = os.path.join(folder_path, output_path)
                
                
        flow_idx = df['flow_idx'][index]
        current_file_packets.append(row)
        packets_written += 1
    
        
# Write any remaining packets to a CSV file
if current_file_packets:
    df_to_write = pd.DataFrame(current_file_packets)
    df_to_write.to_csv(output_path, index=False)

print("Splitting complete.")




