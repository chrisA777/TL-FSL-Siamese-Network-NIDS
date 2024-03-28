# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 18:02:11 2024

@author: Chris
"""



import os
import pandas as pd
import random
from itertools import combinations
import itertools
from sklearn.utils import shuffle
from itertools import product

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

    
# Seperate data into classes - return dict
def seperateClasses(df):
    classes = {}
    for class_label in df['label'].unique():
        classes[class_label] = df[df['label'] == class_label]
    return classes

# Sample similar pairs - Sample 2 examples from grouped data to form pairs
def getSimilar(samples, num_pairs):
    similar_pairs = []
    for i in range(num_pairs):
        similar_pairs.append(samples.sample(2))
        
    return similar_pairs

# Sample dissimilar pairs - Take 1 example from 2 groups and a
def getDissimilar(samples, diff_samples, num_pairs):
    dissimilar_pairs = []
    for i in range(num_pairs):
        sample = samples.sample(1)
        diff_sample = diff_samples.sample(1)
        dissimilar_pairs.append(pd.concat([sample, diff_sample]))
    
    return dissimilar_pairs

def shuffleByPairs(df):  
    groups = [df for _, df in df.groupby('pair')]
    random.shuffle(groups)
    
    df = pd.concat(groups).reset_index(drop=True)   
    
    return df
        
def getTrainingPairs(n_pairs, hidden):
       
    # 50:50 split 
    n_similar = n_pairs // 2
    n_dissimilar = n_similar
    
    n_similar_per_class = n_similar // n_classes
   
    # Empty df to store sampled pairs
    sampled_df = pd.DataFrame()
    
    for df in dfs: 

        # When hidden is true, we are only generating pairs for the hidden class
        if hidden == True:
            if(df['categorical_label'][0] != hidden_class):
                print('skipped')
                continue
            
        print(df.head(1))
        
        # Get number of samples in current class
        n_samples = len(df)
        
        # Train test split - making these equal (allows for more combinations and code reuse)
        n_train = n_samples//2
        
        # Indices beyond this are to be used for training pair
        available_indices = list(range(0, n_train))    
        
        # List of pair indices
        all_pairs = []
        
        # Set to keep track of used indices
        used_combinations = set()
        
        # Sampling similar pairs
        count = 0
        
        # If not dealing with limited class - we have enough samples not to require duplicates
        if hidden == False:
            while (count < n_similar_per_class):
                
                
                sampled_pair = random.sample(available_indices, 2)
                sampled_pair = tuple(sorted(sampled_pair))
                
                if sampled_pair not in used_combinations:
                    all_pairs.append(sampled_pair)
                    count += 1
                    used_combinations.add(sampled_pair)
        
        # For limited class - we will sometimes need duplicates so use alternative method
        else:  
            
            # This line is time consuming so only want to use it when we actually need it
            unique_pairs = list(combinations(available_indices, 2))
            random.shuffle(unique_pairs)
            
            while count < n_similar_per_class and unique_pairs:
                
                sampled_pair = unique_pairs.pop()
                sampled_pair = tuple(sorted(sampled_pair))
                
                all_pairs.append(sampled_pair)
                count += 1
            
            # Now, sample with duplicates if needed
            while count < n_similar_per_class:
    
                sampled_pair = random.choice(available_indices), random.choice(available_indices)
                sampled_pair = tuple(sorted(sampled_pair))
                all_pairs.append(sampled_pair)
                count += 1
        
        while (count < n_similar_per_class):
            
            sampled_pair = random.sample(available_indices, 2)
            sampled_pair = tuple(sorted(sampled_pair))
        
            if sampled_pair not in used_combinations:
                all_pairs.append(sampled_pair)
                count += 1
                used_combinations.add(sampled_pair)
             
        # Adding similar pairs to main df
        for sampled_pair in all_pairs:
            
            # Extract the rows from df based on sampled_pair indices
            index1, index2 = sampled_pair
            selected_rows = df.iloc[[index1, index2], :]
    
            # If sampled_df is empty copy selected rows
            if sampled_df.empty:
                sampled_df = selected_rows.copy()
            else:
                # Ensure columns and order match, then append the rows
                sampled_df = pd.concat([sampled_df, selected_rows], axis=0, ignore_index=True)
                
    print('dissimilar')
    dissimilar_combinations = list(combinations(class_indices, 2))  
    n_combinations = len(dissimilar_combinations)    
    n_dissimilar_per_combination = n_dissimilar // n_combinations
    
    # When only sampling hidden class - logic changes 
    n_dissimilar_hidden = len(sampled_df)/2                                     # Get equal number of dissimilar
    n_combinations_hidden = 0   
    for combination in dissimilar_combinations:                                 # Find number of combinations with hidden class
        if hidden_class_index in combination:
            n_combinations_hidden +=1
            
    print(f'{n_dissimilar_hidden} similar')
    print(f'{n_combinations_hidden} combinations')

    for combination in dissimilar_combinations:
        
        if hidden == True:
            # Change the number of dissimilar per combinations due to changed logic
            n_dissimilar_per_combination = n_dissimilar_hidden / n_combinations_hidden
            
            # Skip combinations without our hidden class
            if hidden_class_index not in combination:

                continue

        # Get the dfs corresponding to the samples of both classes in the combination
        df_1 = dfs[combination[0]]
        df_2 = dfs[combination[1]]
        
        # Get number of samples in each class
        n_samples_1 = len(df_1)
        n_samples_2 = len(df_2)
        
        # Train test split - making these equal (allows for more combinations and code reuse)
        n_train_1 = n_samples_1//2
        n_train_2 = n_samples_2//2
        
        available_indices_1 = list(range(0, n_train_1))
        available_indices_2 = list(range(0, n_train_2))
        
        # List of pair indices
        all_pairs = []
        # Set to keep track of used indices
        used_combinations = set()
        
        # Sampling dissimilar
        count = 0
        while (count < n_dissimilar_per_combination):
            
            sampled_pair = tuple([random.sample(available_indices_1, 1)[0],
                                  random.sample(available_indices_2, 1)[0]])
                        
            if sampled_pair not in used_combinations:
                all_pairs.append(sampled_pair)
                count += 1
                used_combinations.add(sampled_pair)
    
   
        # Adding similar pairs to main df
        for sampled_pair in all_pairs:
            
            # Extract the rows from df based on sampled_pair indices
            index1, index2 = sampled_pair
            selected_row_1 = df_1.iloc[[index1], :] 
            selected_row_2 = df_2.iloc[[index2], :]
    

            sampled_df = pd.concat([sampled_df, selected_row_1, selected_row_2], axis=0, ignore_index=True)

             
    sampled_df['pair'] = (sampled_df.index // 2) + 1 

    return sampled_df


def getHiddenTrainingPairs(n_pairs):
       
    # 50:50 split 
    n_similar = n_pairs // 2
    n_dissimilar = n_similar
    
    n_similar_per_class = n_similar // n_classes
   
    # Empty df to store sampled pairs
    sampled_df = pd.DataFrame()
    
    for df in dfs:   
        print(df.head(1))
        
        if(df['categorical_label'][0] != hidden_class):
            print('skipped')
            continue
        
        # Get number of samples in current class
        n_samples = len(df)
        
        # Train test split - making these equal (allows for more combinations and code reuse)
        n_train = n_samples//2
        
        # Indices beyond this are to be used for training pair
        available_indices = list(range(0, n_train))    
        
        # List of pair indices
        all_pairs = []
        
        # Set to keep track of used indices
        used_combinations = set()
        
        # Sampling similar pairs
        count = 0
        
        while (count < n_similar_per_class):
            
            sampled_pair = random.sample(available_indices, 2)
            sampled_pair = tuple(sorted(sampled_pair))
        
            if sampled_pair not in used_combinations:
                all_pairs.append(sampled_pair)
                count += 1
                used_combinations.add(sampled_pair)
             
        # Adding similar pairs to main df
        for sampled_pair in all_pairs:
            
            # Extract the rows from df based on sampled_pair indices
            index1, index2 = sampled_pair
            selected_rows = df.iloc[[index1, index2], :]
    
            # If sampled_df is empty copy selected rows
            if sampled_df.empty:
                sampled_df = selected_rows.copy()
            else:
                # Ensure columns and order match, then append the rows
                sampled_df = pd.concat([sampled_df, selected_rows], axis=0, ignore_index=True)

    dissimilar_combinations = list(combinations(class_indices, 2))  
    n_combinations = len(dissimilar_combinations)    
    n_dissimilar_per_combinations = n_dissimilar // n_combinations
    
    n_x = len(sampled_df)
    
    for combination in dissimilar_combinations:
        
        if hidden_class_index not in combination:
            n_dissimilar_combinations = n_x//n_combinations
            print('skiiped')
            continue
        # Get the dfs corresponding to the samples of both classes in the combination
        df_1 = dfs[combination[0]]
        df_2 = dfs[combination[1]]
        
        # Get number of samples in each class
        n_samples_1 = len(df_1)
        n_samples_2 = len(df_2)
        
        # Train test split - making these equal (allows for more combinations and code reuse)
        n_train_1 = n_samples_1//2
        n_train_2 = n_samples_2//2
        
        available_indices_1 = list(range(0, n_train_1))
        available_indices_2 = list(range(0, n_train_2))
        
        # List of pair indices
        all_pairs = []
        # Set to keep track of used indices
        used_combinations = set()
        
        # Sampling dissimilar
        count = 0
        while (count < n_dissimilar_per_combinations):
            sampled_pair = tuple([random.sample(available_indices_1, 1)[0],
                                  random.sample(available_indices_2, 1)[0]])
                        
            if sampled_pair not in used_combinations:
                all_pairs.append(sampled_pair)
                count += 1
                used_combinations.add(sampled_pair)
    
   
        # Adding similar pairs to main df
        for sampled_pair in all_pairs:
            
            # Extract the rows from df based on sampled_pair indices
            index1, index2 = sampled_pair
            selected_row_1 = df_1.iloc[[index1], :] 
            selected_row_2 = df_2.iloc[[index2], :]
    

            sampled_df = pd.concat([sampled_df, selected_row_1, selected_row_2], axis=0, ignore_index=True)

             
    sampled_df['pair'] = (sampled_df.index // 2) + 1

    groups = [df for _, df in sampled_df.groupby('pair')]
    random.shuffle(groups)
    
    sampled_df = pd.concat(groups).reset_index(drop=True)    

    return sampled_df
        
        
      
def getValidationPairs(n_pairs):
    
    # Each test sample must be pairs with a sample of each class
    n_test_samples = n_pairs // n_classes
    n_test_samples_per_class = n_test_samples // n_classes

    # Empty list to store selected rows
    sampled_rows = []
    
    for df in dfs:
        n_test_samples_per_class = 3000
        

    
        # Get number of samples in current class
        n_samples = len(df)
        
        print(n_samples)
        
        # Train test split - making these equal (allows for more combinations and code reuse)
        n_train = n_samples // 2
        
        # Indices before this are to be used for training pair
        available_indices = list(range(n_train, n_samples))
        
        # Get indices of each test sample
        test_samples = random.choices(available_indices, k=n_test_samples_per_class)
        
        count = 0
        
        # For each of our test samples
        for test_sample in test_samples:
            count += 1
            
            if count % 100 == 0:
                print(count)
    
            # Find a train sample from each class to pair it with
            for _ in range(5):
                for train_df in dfs:
                    # Pair our test sample with a sample from the training range
                    n_train_samples = len(train_df) // 2
                    train_sample = random.randint(0, n_train_samples - 1)
                        
                    selected_row_1 = df.iloc[test_sample]
                    selected_row_2 = train_df.iloc[train_sample]
                    
                    # Append selected rows to the list
                    sampled_rows.extend([selected_row_1, selected_row_2])
    
    # Concatenate all collected rows at once
    sampled_df = pd.DataFrame(sampled_rows)
    sampled_df = sampled_df.reset_index(drop = True)
           
                        
    sampled_df['pair'] = (sampled_df.index // 2) + 1
    sampled_df['sample'] = (sampled_df.index // (n_classes*2*5)) + 1

    sampled_df['y_true'] = None
    
    # True value is always in the first packet in pair
    for i in range(0, len(sampled_df), 2):
        label = sampled_df.at[i, 'categorical_label']
        sampled_df.at[i, 'y_true'] = label
        sampled_df.at[i+1, 'y_true'] = label

    groups = [df for _, df in sampled_df.groupby('sample')]
    
    random.shuffle(groups)
    

    sampled_df = pd.concat(groups).reset_index(drop=True)
        
    return sampled_df
    
        
if __name__ == "__main__":
    
    # Read data for all classes
    dfs, classes = getData()
    
    n_classes = len(classes)
    class_indices = list(range(n_classes))
    
    """
    # Define hidden class and its index
    hidden_class = 'hulk'
    hidden_class_index = classes.index(hidden_class)
    hidden_class_folder = os.path.join('## Limited Data', hidden_class + '_2')
    
    n_train_pairs = 30e3
    
    # Get training pairs with hidden as False - this will not hide any classes
    train_pairs = getTrainingPairs(n_train_pairs, hidden = False)

    # Now lets seperate the hidden class to form original training set
    grouped = train_pairs.groupby('pair')
    filtered_pairs = pd.DataFrame(columns=train_pairs.columns)
    
    for name, group in grouped:
        if hidden_class not in group['categorical_label'].values:
            filtered_pairs = pd.concat([filtered_pairs, group])
    
    filtered_pairs = shuffleByPairs(filtered_pairs)
    
    # This is our base file 
    filtered_pairs.to_csv((os.path.join(hidden_class_folder,
                                       f'base_file_{hidden_class}.csv')), index=False)

    
    # Define number of samples we want to test - must be in descending order
    n_attack_samples = list(range(200, 10, -20))
    n_attack_samples.append(10)
    n_attack_samples.append(8)
    n_attack_samples.append(4)
    n_attack_samples.append(2)
    
    # Iterate through each
    for n in n_attack_samples:
        dfs[hidden_class_index] = dfs[hidden_class_index].sample(n)             # Replace original hidden df with one with limited samples 
        dfs[hidden_class_index] = dfs[hidden_class_index].reset_index(
            drop=True, inplace = False)                                         # Reset index
        hidden_pairs = getTrainingPairs(n_train_pairs, hidden = True)           # Now call get training pairs with hidden True, and getting only pairs with hidden class
        hidden_pairs = shuffleByPairs(hidden_pairs)
        full_df = pd.concat([filtered_pairs, hidden_pairs])                     # Concat to get full df
        
        hidden_pairs.to_csv((os.path.join(hidden_class_folder,
                                         f'{hidden_class}_{n}.csv')), index = False)
        full_df.to_csv((os.path.join(hidden_class_folder,
                                    f'{hidden_class}_complete_{n}.csv')), index = False)
    """  
    val_pairs = getValidationPairs(5000)
    val_pairs.to_csv('Validation_Pairs_2000.csv', index = False)
    
    
            
    
    


                
            
    
    
    
    
    
    

    



