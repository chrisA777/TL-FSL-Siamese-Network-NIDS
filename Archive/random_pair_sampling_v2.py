# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 09:47:01 2024

@author: Chris

Description: Code to randomly sample dataset for input pairs

Using random_sampling.txt method from Jack
"""

import os
import pandas as pd
import random
from itertools import combinations
import itertools
from sklearn.utils import shuffle
from itertools import product

# Get df 
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
 
        
def getTrainingPairs(n_pairs):
       
    # 50:50 split 
    n_similar = n_pairs // 2
    n_dissimilar = n_similar
    
    n_similar_per_class = n_similar // n_classes
   
    # Empty df to store sampled pairs
    sampled_df = pd.DataFrame()
    
    for df in dfs:   
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
        
        if df['categorical_label'][0] == 'benign':
            
            # Now, sample with duplicates if needed
            while count < n_similar_per_class:
    
                sampled_pair = random.choice(available_indices), random.choice(available_indices)
                sampled_pair = tuple(sorted(sampled_pair))
                all_pairs.append(sampled_pair)
                count += 1
                
        else: 
            
            unique_pairs = list(combinations(available_indices, 2))
            random.shuffle(unique_pairs)
            
            print(available_indices)
            print('*' * 50)
            
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
    
    for combination in dissimilar_combinations:
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

        print(available_indices_1)
        print('*' * 50)
        print(available_indices_2)
        # List of pair indices
        all_pairs = []
        # Set to keep track of used indices
        used_combinations = set()
        
        # Sampling dissimilar
        count = 0
        
        unique_pairs = list(product(available_indices_1, available_indices_2))
        random.shuffle(unique_pairs)
        

        
        while count < n_dissimilar_per_combinations and unique_pairs:
            
            sampled_pair = unique_pairs.pop()
            sampled_pair = tuple(sampled_pair)
            all_pairs.append(sampled_pair)
            count += 1

        # Now, sample with duplicates if needed
        while count < n_dissimilar_per_combinations:
            
            if count % 10 == 0:
                print(count)

            sampled_pair = random.choice(available_indices_1), random.choice(available_indices_2)
            sampled_pair = tuple(sampled_pair)
            all_pairs.append(sampled_pair)
            count += 1
    
   
        # Adding similar pairs to main df
        for sampled_pair in all_pairs:
            
            # Extract the rows from df based on sampled_pair indices
            index1, index2 = sampled_pair
            selected_row_1 = df_1.iloc[[index1], :] 
            selected_row_2 = df_2.iloc[[index2], :]
    

            sampled_df = pd.concat([sampled_df, selected_row_1, selected_row_2], axis=0, ignore_index=True)

    return sampled_df
        
      
def getValidationPairs(n_pairs):
    
    # Each test sample must be pairs with a sample of each class
    n_test_samples = n_pairs // n_classes
    n_test_samples_per_class = n_test_samples // n_classes

    # Empty df which will be used to store rows
    sampled_df = pd.DataFrame()
    

    for df in dfs:
        
        if (df['categorical_label'][0] == 'benign'):
            n_test_samples_per_class = 5000
        else:
            n_test_samples_per_class = 80

        # Get number of samples in current class
        n_samples = len(df)
        
        print(n_samples)
        
        # Train test split - making these equal (allows for more combinations and code reuse)
        n_train = n_samples//2
        
        # Indices before this are to be used for training pair
        available_indices = list(range(n_train, n_samples))
        
        print(available_indices)
        print(n_test_samples_per_class)
        
        # Get indices of each test sample
        test_samples = random.sample(available_indices, n_test_samples_per_class)
        
        count =0
        
        # For each of our test samples
        for test_sample in available_indices:
            count+=1
            if (count%10 == 0):
                print(count)
                print("x")
            # Find a train sample from each class to pair it with
            for _ in range(5):
                   
                for train_df in dfs:
                    # Pair our test sample with a sample from the training range
                    n_train_samples = len(train_df) // 2
                    train_indices = list(range(0, n_train_samples))
                    
                    train_sample = random.sample(train_indices, 1)[0]
                        
                    selected_row_1 = df.iloc[[test_sample], :]
                    selected_row_2 = train_df.iloc[[train_sample], :] 
                    
                    # If sampled_df is empty copy selected rows
                    if sampled_df.empty:
                        sampled_df = selected_row_1.copy()
                        sampled_df = pd.concat([sampled_df, selected_row_2], axis=0, ignore_index=True)
                    else:
                        # Ensure columns and order match, then append the rows
                        sampled_df = pd.concat([sampled_df, selected_row_1, selected_row_2], axis=0, ignore_index=True) 
        
    return sampled_df
    
        
if __name__ == "__main__":
      
    # Get a list of dfs - one for each class - now specifying the number of attack samples
    # for each class 
    
    ns = list(range(200, 0, -10))
    ns.append(8)
    ns.append(4)
    ns.append(2)


    for n_attack_samples in ns:

        dfs, classes = getData()
        
        for i in range(len(dfs)):
            if dfs[i]['categorical_label'][0] == 'benign':
                dfs[i] = dfs[i].sample(10000, random_state = 72).reset_index(
                                            drop=True, inplace = False) 
            else:
                dfs[i] = dfs[i].sample(n_attack_samples, random_state = 72).reset_index(
                                            drop=True, inplace = False) 
        
        n_classes = len(classes)
        class_indices = list(range(n_classes))  # numerate class list
           
        train_file_name = f'Training_Pairs_{n_attack_samples}.csv'
        val_file_name = f'Validation_Pairs_{n_attack_samples}.csv'
        
        train_pairs = getTrainingPairs(30e3)
        #val_pairs = getValidationPairs( n_val_pairs)
             
        train_pairs['pair'] = (train_pairs.index // 2) + 1
        #val_pairs['pair'] = (val_pairs.index // 2) + 1
        #val_pairs['sample'] = (val_pairs.index // (n_classes*2*5)) + 1
    
        #val_pairs['y_true'] = None
        
        # True value is always in the first packet in pair
        #for i in range(0, len(val_pairs), 2):
            #label = val_pairs.at[i, 'categorical_label']
            #val_pairs.at[i, 'y_true'] = label
           # val_pairs.at[i+1, 'y_true'] = label

        train_groups = [df for _, df in train_pairs.groupby('pair')]
        #val_groups = [df for _, df in val_pairs.groupby('sample')]
        
        random.shuffle(train_groups)
        #random.shuffle(val_groups)
        
        train_pairs = pd.concat(train_groups).reset_index(drop=True)
        #val_pairs = pd.concat(val_groups).reset_index(drop=True)
        
        train_pairs.to_csv(os.path.join('## Limited Data', 'All_2', train_file_name), index = False)
        #val_pairs.to_csv(val_file_name, index = False)
        
                
        
    
    
    
    
    
    
    

    



