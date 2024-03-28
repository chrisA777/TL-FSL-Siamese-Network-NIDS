# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 23:30:08 2023

@author: Chris
"""

import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.optimizers.legacy import SGD
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.utils import to_categorical

def set_rand_seed(seed_value):
    
    # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed_value)
    
    # 2. Set the `python` built-in pseudo-random generator at a fixed value
    import random
    random.seed(seed_value)
    
    # 3. Set the `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)
    
    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)
    # for later versions: 
    tf.compat.v1.set_random_seed(seed_value)

def getData(csv_name):

        
    from sklearn.utils import shuffle
    
    # Read in training dataset and separate features and target
    df = pd.read_csv(csv_name)
    
    X = df[selected_features]
    y = df['categorical_label']
    
    # Shuffle X and y together
    X, y = shuffle(X, y, random_state=42)  # You can adjust the random_state if needed
    
    # Apply standard scalar to features
    column_names = X.columns.tolist()  # Store feature names first 
    scaler = StandardScaler()
    X = scaler.fit_transform(X)  # Scaler changes original column names (undesirable)
    X = pd.DataFrame(X, columns=column_names)  # Adding original column names
    
    # Encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y)
    y_encoded = encoder.transform(y)
    # convert integers to dummy variables (i.e. one hot encoded)
    y_one_hot = to_categorical(y_encoded)
    
    y_labels = y
    
    # Create a dictionary mapping numerical values to categorical labels
    label_mapping = {num: label for num, label in zip(y_encoded, y)}
    
    return X, y_one_hot, y_encoded, label_mapping

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

def build_model(n_inputs):
    
    optimiser = Adam(0.001)
    
    # Defining a keras models with 3 layers - 2 hidden + 1 output
    # Note: need to redefine this each time
    model = Sequential()                                                        
    model.add(Dense(128,  input_shape = (n_features, ),
                    activation = 'relu', kernel_regularizer=l2(lambdal2)))
    model.add(Dense(64, activation = 'relu', activity_regularizer=l2(lambdal2)))
    model.add(Dense(32, activation = 'relu', activity_regularizer=l2(lambdal2)))                                              
    model.add(Dense(5, activation = 'softmax'))      
                           
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=optimiser, 
                  metrics=['accuracy'])
    
    return model

def fit_model(X_train, y_train, X_test, y_test):
    
    # Get our model
    model = build_model(n_features)
    
    # Number of epochs to iterate through
    n_epochs = 200
    
    # Fit the model to the data and record training history
    history = model.fit(X_train, y_train, 
                        epochs = n_epochs, batch_size = 5000, 
                        validation_data = (X_test, y_test))
    
    return model, history

def plot_confusion_matrix(y_true, y_pred, title, save_path = None):
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels = class_names)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False, 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(title)
    
    #if save_path:
     #   plt.savefig(save_path)
        
    plt.show()
    

# Set fixed seed for reproducibility
seed_value = 1
set_rand_seed(seed_value)

# Define the number of inputs
n_features = 10

# Select n features 
selected_features = feature_selection(n_features)

test_size = 5000

precisions = []
recalls = []
f1s = []

# Get training and validation data
X, y, y_encoded, label_mapping = getData('sampled_data.csv')


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                    random_state=seed_value, stratify = y)


lambdal2= 0.01

# Fit our model and get training history
model, history = fit_model(X_train, y_train, X_test, y_test)


# Make our predictions using the model

# Convert one hot to numerical
y_test = np.argmax(y_test, axis=1)

# Make predictions
y_pred = model.predict(X_test)

# Convert predictions to numerical labels
y_pred = np.argmax(y_pred, axis=1)

# Generate classification report
report_dict = classification_report(y_test, y_pred, output_dict=True)

# Extract precision, recall, and f1-score for each class
precision = [report_dict[str(i)]['precision'] for i in range(len(report_dict) - 3)]  # Exclude 'accuracy', 'macro avg', and 'weighted avg'
recall = [report_dict[str(i)]['recall'] for i in range(len(report_dict) - 3)]
f1_score = [report_dict[str(i)]['f1-score'] for i in range(len(report_dict) - 3)]

precisions.append(report_dict['weighted avg']['precision'])
recalls.append(report_dict['weighted avg']['recall'])
f1s.append(report_dict['weighted avg']['f1-score'])


    
    
# Get class names from label_mapping in order of sorted keys
sorted_keys = sorted(label_mapping.keys())
class_names = [label_mapping[key] for key in sorted_keys]

y_pred = [class_names[idx] for idx in y_pred]
y_test = [class_names[idx] for idx in y_test]

class_names = ['benign', 'FTP', 'hulk', 'slowloris', 'SSH']

# Define a custom sorting key function
def sort_key(item):
    return class_names.index(item)

# Sort y_pred and y_test based on the order of class_names
y_pred = sorted(y_pred, key=sort_key)
y_test   = sorted(y_test, key=sort_key)


plot_confusion_matrix(y_test, y_pred, 'Confusion Matrix')

# Create a bar chart for precision, recall, and f1-score
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Bar chart for precision
axs[0].bar(class_names, precision, color='blue')
axs[0].set_title('Precision')

# Bar chart for recall
axs[1].bar(class_names, recall, color='green')
axs[1].set_title('Recall')

# Bar chart for f1-score
axs[2].bar(class_names, f1_score, color='orange')
axs[2].set_title('F1-Score')

# Rotate x-axis labels for better visibility
for ax in axs:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()



# Get the average training and validation metrics
avg_train_accuracy = history.history['accuracy']
avg_train_loss = history.history['loss']


avg_val_accuracy = history.history['val_accuracy']
avg_val_loss = history.history['val_loss']

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 8))

# Set title for the entire figure
fig.suptitle('Loss & Accuracy Curves - Introduction of L2 Regularisation', fontsize=18)

# Plot training and validation loss from history
axs[0].plot(avg_train_loss, label='Training Loss - No Regularisation')
axs[0].plot(avg_val_loss, label='Validation Loss - No Regularisation')

axs[0].set_xlabel('Epochs', fontsize=14)
axs[0].set_ylabel('Loss', fontsize=14)
axs[0].legend(fontsize=12)
axs[0].tick_params(axis='both', which='major', labelsize=12)

# Plot training and validation accuracy from history
axs[1].plot(avg_train_accuracy, label='Training Accuracy - No Regularisation')
axs[1].plot(avg_val_accuracy, label='Validation Accuracy - No Regularisation')


axs[1].set_xlabel('Epochs', fontsize=14)
axs[1].set_ylabel('Accuracy', fontsize=14)
axs[1].legend(fontsize=12)
axs[1].tick_params(axis='both', which='major', labelsize=12)

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()

# Obtaining min and max metrics for printing
min_loss_index = np.argmin(avg_val_loss)
max_accuracy_index = np.argmax(avg_val_accuracy)



print('min loss: {} at index {}'.format(np.min(avg_val_loss), min_loss_index))
print('max accuracy: {} at index {}'.format(np.max(avg_val_accuracy), max_accuracy_index))







