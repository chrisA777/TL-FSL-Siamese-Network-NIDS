
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 12:24:36 2024

@author: Chris
"""

# General
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mode
import csv


import optuna
from optuna_dashboard import run_server
from optuna.samplers import RandomSampler

# Keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import LearningRateScheduler

# Scikit 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier

# Guarantees reproduction of results - NOT MY CODE
def set_rand_seed(seed_value):
    
    # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed_value)
    
    # 2. Set the `python` built-in pseudo-random generator at a fixed value
    import random
    random.seed(seed_value)
    
    # 3. Set the `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)
    
    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    #tf.random.set_seed(seed_value)
    # for later versions: 
    tf.compat.v1.set_random_seed(seed_value)

# Apply RFE on blackbox model - this isn't being applied to our siamese model
def feature_selection(n_features):
    
    # The project only focuses on 4 attack classes
    classes = ['benign', 'slowloris', 'hulk', 'SSH', 'FTP']
    n_samples = 2000  # Number of samples from each class
    
    data_folder = os.path.join(os.getcwd(), '## Data')
    
    # Concatenate data from all classes - get 2000 samples from each
    dfs = []
    
    for class_name in classes:
        df = pd.read_csv(os.path.join(data_folder, class_name + '.csv'))
        df = df.fillna(0)
        dfs.append(df.sample(n=n_samples))
        
    df_combined = pd.concat(dfs).sample(frac=1).reset_index(drop=True)
    
    X = df_combined.drop(columns=['label', 'categorical_label'])
    y = df_combined['categorical_label']
    
    # Encode categorical labels
    label_encoder = LabelEncoder()
    y_x = label_encoder.fit_transform(y)
    
    # Apply RFE
    estimator = DecisionTreeClassifier(random_state=42)
    selector = RFE(estimator, n_features_to_select=n_features)
    selector = selector.fit(X, y_x)
    selected_features = X.columns[selector.support_]
    
    return selected_features
    
    
def getData(csv):
    
    # Define folder paths, file names, etc
    cwd = os.getcwd()
    #dataset_folder = os.path.join(cwd, "00 Datasets")
    csv_path = os.path.join(csv)
    
    # Read in training dataset and seperate features and target
    df = pd.read_csv(csv_path)
    
    X = df[selected_features]
    y = df['categorical_label']
    
    n_classes = y.nunique()
    
    # y true only exists for val
    if 'y_true' not in df.columns:
        df['y_true'] = None
        
    # Take when index%nclasses == 0
    y_true = df.loc[df.index % (n_classes*2) == 0, 'y_true']
    y_true.reset_index(drop=True, inplace=True)

    global scaler  # Access the global scaler object
    
    # If scaler is not fitted yet, fit it
    if scaler is None:
        column_names = X.columns.tolist()
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
    else:
        column_names = X.columns.tolist()
        X = scaler.transform(X)
    
    X = pd.DataFrame(X, columns=column_names)                              
    
    # Separate even and odd indices for X and y
    X_even = X[::2]
    X_odd = X[1::2]
    y_even = y[::2].tolist()
    y_odd = y[1::2].tolist()
    
    # 1 for similar and 0 for dissimilar
    y_similar = [int(y_even[i] == y_odd[i]) for i in range(len(y_even))]
    y_similar = pd.Series(y_similar)
    
    return X_even, X_odd, X, y, y_similar, y_true


def EuclDistance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))
    
def EuclDistanceShape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)
    

def accuracy(y_true, y_pred):
    

    # Check if predicted values are less than 0.5
    thresholded_preds = y_pred < 0.5
    
    # Cast boolean results to the same type as y_true
    thresholded_preds = K.cast(thresholded_preds, y_true.dtype)
    
    # Compare true labels with the casted predicted values
    correct_predictions = K.equal(y_true, thresholded_preds)
    
    # Mean of boolean tensor
    accuracy = K.mean(correct_predictions)

    return accuracy

def plot_confusion_matrix(y_true, y_pred, title, save_path = None):
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels = classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False, 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()
    

# Save model callback - saves trained model with best params on the go
def save_callback(study,trial):
    if study.best_trial == trial:
        model.save_weights(best_model_name)
        
        # Initialize an empty dictionary to store parameters and their values
        params_dict = {"Score": trial.value, "Seed Value": seed_value}
        
        # Add trial parameters to the dictionary
        for key, value in trial.params.items():
            params_dict[key] = value
        
        # Create DataFrame from the dictionary
        df = pd.DataFrame(params_dict, index=[0])
        
        # Write DataFrame to CSV file
        df.to_csv(best_csv_name, index=False)


# Multi-purpose function - for optimising hyperparameters using Optuna & for returning initial trained
def train_objective(trial):
    
    global model
    
    if trial:
    
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
        
        batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024, 2048, 5000, 10000, 15000, 30000])
        n_epochs = trial.suggest_int('n_epochs', 10, 300)
        #n_epoch_2 = trial.suggest_int('n_epochs_2', 10, 300)
        # Network Architecture
        n_layers = trial.suggest_int('n_layers', 1, 4) 
        nodes_per_layer = [trial.suggest_categorical(f'n_nodes_{i}',
                                                     [8 , 16, 32, 64, 128, 256, 512]) for i in range(n_layers)]  
        d_out = trial.suggest_categorical('d_out', [8, 16, 32, 64, 128, 256, 512])
        
        dropout = trial.suggest_discrete_uniform('dropout', 0.1, 0.6, 0.1)
        margin = trial.suggest_discrete_uniform('margin', 0.0, 1.0, 0.1)
        
        # Cosine Annealing
        min_val = trial.suggest_loguniform('min_val', 1e-8, 1e-5)  
        
        max_val = trial.suggest_uniform('max_val', 0.001, 0.01)  
        
    else:
        # Load parameters from CSV
        params_df = pd.read_csv(best_csv_name)
        
        # Assign parameters
        weight_decay = params_df['weight_decay'].iloc[0]
        batch_size = params_df['batch_size'].iloc[0]
        n_epochs = params_df['n_epochs'].iloc[0]
        n_layers = int(params_df['n_layers'].iloc[0])
        nodes_per_layer = [params_df[f'n_nodes_{i}'].iloc[0] for i in range(n_layers)]
        d_out = params_df['d_out'].iloc[0]
        dropout = params_df['dropout'].iloc[0]
        margin = params_df['margin'].iloc[0]
        min_val = params_df['min_val'].iloc[0]
        max_val = params_df['max_val'].iloc[0]
        
    weight_decay = round(weight_decay, 3)
    min_val = round(min_val, 3)
    max_val = round(max_val, 3)
        
        
    def cosine_annealing(epoch, T_max, min_val, max_val):
        return min_val + 0.5 * (max_val - min_val) * (1 + np.cos((epoch * np.pi) / T_max))

    def ContrastiveLoss(y_true, y_pred, margin=margin):
        y_true = tf.cast(y_true, tf.float32)
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

    def getModel(nodes_per_layer, n_features):
        input_layer = Input(shape=(n_features,), name='Input_Layer')
        x = input_layer
        for nodes in nodes_per_layer:
            x = Dense(nodes, activation='relu')(x)
            x = Dropout(dropout)(x)
        x = Dense(d_out)(x)
        return Model(inputs=input_layer, outputs=x)

    def SiameseNetwork(n_features):
        base_model = getModel(nodes_per_layer, n_features)
        input_1 = Input(shape=(n_features,), name='input_1of2')
        vect_output_1 = base_model(input_1)
        input_2 = Input(shape=(n_features,), name='input_2of2')
        vect_output_2 = base_model(input_2)
        output = Lambda(EuclDistance, name='output_layer', output_shape=EuclDistanceShape)([vect_output_1, vect_output_2])
        model = Model([input_1, input_2], output)
        return model
    
    model = SiameseNetwork(n_features)
    
    if trial:    
        lr_scheduler = LearningRateScheduler(lambda epoch: cosine_annealing(epoch, n_epochs, min_val, max_val))
        adamW = AdamW(learning_rate=max_val, weight_decay=weight_decay)
        model.compile(loss=ContrastiveLoss, optimizer=adamW)
    
        history = model.fit([x1_train, x2_train], y_similar, epochs=n_epochs, batch_size=batch_size, callbacks=[lr_scheduler],
                            verbose=0)
        
    else:
        model.load_weights(best_model_name)
        
    normal_predictions = make_predictions(model)
    accuracy = accuracy_score(y_true, normal_predictions)
     
    # If optimising return accuracy metric for optuna
    if trial:
        return accuracy
    # Else return the trained model
    else:
        
        title = f'Confusion Matrix After Initial Training - {limited_class} Excluded'
        plot_confusion_matrix(y_true, normal_predictions, title, best_img_name)
        print(accuracy)
        return model

# Function for retraining our model with class e included 
def retrain_objective(trial):
    
    global model
    
    if trial:

        weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
        
        batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024, 2048, 5000, 10000, 15000, 30000])
        n_epochs = trial.suggest_int('n_epochs', 10, 300)
        n_epochs_2 = trial.suggest_int('n_epochs_2', 10, 300)

        
        margin = trial.suggest_discrete_uniform('margin', 0.0, 1.0, 0.1)
        
        # Cosine Annealing
        min_val = trial.suggest_loguniform('min_val', 1e-8, 1e-5)  # Suggest min_val
        max_val = trial.suggest_uniform('max_val', 0.001, 0.01)  # Suggest max_val
        
    else:
        
        # Load parameters from CSV
        params_df = pd.read_csv(best_csv_name)
        
        # Assign parameters
        weight_decay = params_df['weight_decay'].iloc[0]
        batch_size = params_df['batch_size'].iloc[0]
        n_epochs = params_df['n_epochs'].iloc[0]
        margin = params_df['margin'].iloc[0]
        min_val = params_df['min_val'].iloc[0]
        max_val = params_df['max_val'].iloc[0]
        
    weight_decay = round(weight_decay, 3)
    min_val = round(min_val, 3)
    max_val = round(max_val, 3)
         
    def cosine_annealing(epoch, T_max, min_val, max_val):
        return min_val + 0.5 * (max_val - min_val) * (1 + np.cos((epoch * np.pi) / T_max))

    def ContrastiveLoss(y_true, y_pred, margin=margin):
        y_true = tf.cast(y_true, tf.float32)
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
    
    model = original_model
    
    if trial:
        lr_scheduler = LearningRateScheduler(lambda epoch: cosine_annealing(epoch, n_epochs, min_val, max_val))
        adamW = AdamW(learning_rate=max_val, weight_decay=weight_decay)
        model.compile(loss=ContrastiveLoss, optimizer=adamW)
    
        history = model.fit([x1_train, x2_train], y_similar, epochs=n_epochs, batch_size=batch_size, callbacks=[lr_scheduler],
                            verbose=0)
    else:

        model.load_weights(best_model_name)


    normal_predictions = make_predictions(model)
    accuracy = accuracy_score(y_true, normal_predictions)
    
    
    if trial:
        return accuracy
    else:
        
        title = f'Confusion Matrix After Retraining - {limited_class} Excluded'
        plot_confusion_matrix(y_true, normal_predictions, title, best_img_name)
        report_dict = classification_report(y_true, normal_predictions,
                                                   output_dict=True)

        print(accuracy)
        return model, report_dict
    

    
def make_predictions(model):
    distances = model.predict([x1_val, x2_val])
    
    # Determine the number of test samples - one for each class
    n_samples = len(distances) // len(classes)
    
    # Reshape
    distances = distances.reshape((n_samples, len(classes)))
    
    # Initialize list to store minimums of every 5 lists
    labels = []
    # Iterate over every 5 lists
    for i in range(0, len(distances), 5):
        # Get 5 lists
        five_lists = distances[i:i+5]
        
        # Find the index of the minimum in each list
        min_indices = np.argmin(five_lists, axis=1)
        
        # Find the most occurring minimum index
        most_common_min_index = mode(min_indices)
        
        # Append the most occurring minimum index to labels
        labels.append(classes[most_common_min_index])
    
    return labels


# Set fixed seed for reproducibility
"""
-> Random seeds used for each limited class
-> Swapped around becauase random sampler did badly with some seeds 
   
   SSH 8
   FTP 7
   slowloris 64
   hulk 99
   
"""

seed_value = 8
set_rand_seed(seed_value) 

# Select limited class & n features & n attack samples
limited_class = 'SSH'
n_features = 16
n_attacks = 5

# Select n features - 16 found to do well
selected_features = feature_selection(n_features)


limited_class_path = os.path.join('## Limited Data', limited_class )

base_file= os.path.join(limited_class_path,
                        f'base_file_{limited_class}.csv')
full_file = os.path.join(limited_class_path,
                         f'{limited_class}_complete_{n_attacks*2}.csv')         # Multiply by 2 as file naming in pair sampling took n as n*2


scaler = None

# Get training and validation data with only selected features
x1_train, x2_train, X_train, y_train,  y_similar, _ = getData(base_file)
x1_val, x2_val, _, y,  y_val, y_true = getData(f'Validation_Pairs_{limited_class}.csv')

y_true = y[::40]

# Get list of classes 
classes = list(y.iloc[1::2].unique())

# Define fila names for best model and best params
best_model_name = f'{limited_class}_initial.keras'
best_model_name = os.path.join(limited_class_path, best_model_name)
best_csv_name = f'{limited_class}_initial.csv'
best_csv_name = os.path.join(limited_class_path, best_csv_name)
best_img_name = f'{limited_class}_initial.png'
best_img_name = os.path.join(limited_class_path, best_img_name)

"""
COMMENT IN & OUT AS NEEDED (WHEN OPTIMISING - ELSE USE PRE TRAINED )
# Optisation for first set of hyperparameters for initial training - uncomment if needed
storage = optuna.storages.InMemoryStorage()
sampler = RandomSampler(seed=seed_value)
study = optuna.create_study(direction='maximize',sampler=sampler, storage = storage)
study.optimize(train_objective, n_trials=20, callbacks = [save_callback])
best_params = study.best_params
best_loss = study.best_value
#run_server(storage)
"""

# Get saved model from best run for initial training - set trial to None
original_model = train_objective(0)
    
# Need to reset our scaler
scaler = None

# Get training and validation data with only selected features
x1_train, x2_train, X_train, y_train,  y_similar, _ = getData(full_file)
x1_val, x2_val, _, y,  y_val, y_true = getData(f'Validation_Pairs_v2_Equal.csv')

y_true = y[::50]

# Get list of classes 
classes = list(y.iloc[1::2].unique())

# Redfine our file names for 2nd run of hyperparameter optimisation
best_model_name = f'{limited_class}_retrained.keras'
best_model_name = os.path.join(limited_class_path, best_model_name)
best_csv_name = f'{limited_class}_retrained.csv'
best_csv_name = os.path.join(limited_class_path, best_csv_name)
best_img_name = f'{limited_class}_retrained.png'
best_img_name = os.path.join(limited_class_path, best_img_name)

retrained_model, report = retrain_objective(0)

"""
COMMENT IN & OUT AS NEEDED (WHEN OPTIMISING - ELSE USE PRE TRAINED )
# Run hyperparameter optimisation to find best params for retraining
storage = optuna.storages.InMemoryStorage()
sampler = RandomSampler(seed=7)
study = optuna.create_study(direction='maximize',sampler=sampler, storage = storage)
study.optimize(retrain_objective, n_trials=30, callbacks = [save_callback])
best_params = study.best_params
best_score = study.best_value
run_server(storage)
"""



"""
 
SIAMESE KNN WAS NOT USED IN THE END BUT INCLUDED FOR REFERENCE

def SiameseNetworkKNN():
    
    base_model = getModel()

    input_1 = Input(shape=(n_features,), name = 'input_1of2')                   # Only need a single network for applying kNN
    vect_output_1 = base_model(input_1)                                         # Add input layer to model

    model = Model(input_1 , vect_output_1)                                       # One input - One output vector (embdedded space)
    
    
    return model

def make_predictions_KNN():
    
    # Data for fitting KNN is different in its structure
    KNN_x_train = X_train
    KNN_y_train = y_train
    
    KNN_x_val = x1_val[::25] .reset_index(drop=True)                            # Our test sample is in every 25th i of even rows
    KNN_y_val = y[::50].reset_index(drop=True)                                  # Our test sample label is in every 50th i of all rows
    
    model_KNN = SiameseNetworkKNN()
    model_KNN.load_weights('test.keras')
    
    embeddings_train = model_KNN.predict(KNN_x_train)[0]
      
    knn = KNeighborsClassifier(n_neighbors = len(classes))
    knn.fit(embeddings_train, KNN_y_train)
    
    x_val = x1_val[::25]
    
    embeddings_val = model_KNN.predict(KNN_x_val)[0]
    
    labels = knn.predict(embeddings_val)
    
    return labels

"""



