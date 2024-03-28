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
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier

# Guarantees reproduction of results - not my code
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

# Apply RFE on blackbox model
def feature_selection(n_features):
    
    classes = ['benign', 'slowloris', 'hulk', 'SSH', 'FTP']
    n_samples = 2000  # Number of samples from each class
    
    data_folder = os.path.join(os.getcwd(), '## Data')
    
    # Concatenate data from all classes
    dfs = []
    
    for class_name in classes:
        df = pd.read_csv(os.path.join(data_folder, class_name + '.csv'))
        df = df.fillna(0)
        dfs.append(df.sample(n=n_samples))
    df_combined = pd.concat(dfs).sample(frac=1).reset_index(drop=True)
    
    X = df_combined.drop(columns=['label', 'categorical_label'])
    y = df_combined['categorical_label']
    
    label_encoder = LabelEncoder()
    y_x = label_encoder.fit_transform(y)
    
    estimator = DecisionTreeClassifier(random_state=42)
    selector = RFE(estimator, n_features_to_select=n_features)
    selector = selector.fit(X, y_x)
    selected_features = X.columns[selector.support_]
    
    return selected_features
    
    
scaler = None
    
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

def ContrastiveLoss(y_true, y_pred, margin = 0.7):
    
    y_true = tf.cast(y_true, tf.float32)

    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)

# Defining base ('common') model
def getModel():
    
    input_layer = Input(shape=(n_features,), name = 'Input_Layer')               # Create input layer dependent on n_features                        
    x = Dense(64, activation='relu' )(input_layer)  # Connect Dense layer to input
    x = Dropout(0.1)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(8)(x)

    return Model(inputs=input_layer, outputs=x)                                 # Return model with input and output as opposed to just a group of layers

# Quite a nice way to visualise network structure
#def plotNetwork(model):
    
    #from keras.utils.vis_utils import plot_model
    #plot_model(model, to_file='siamese_network.png', show_shapes=True)


def SiameseNetwork():
    
    base_model = getModel()

    input_1 = Input(shape=(n_features,), name = 'input_1of2')                   # Input layer for example 1 in pair 
    vect_output_1 = base_model(input_1)                                         # Add input layer to model

    input_2 = Input(shape=(n_features,), name = 'input_2of2')                   # Input layer for example 2 in pair
    vect_output_2 = base_model(input_2)
    
    output = Lambda(EuclDistance, name='output_layer',
                    output_shape = EuclDistanceShape)([vect_output_1, vect_output_2])
    
    model = Model([input_1, input_2], output)                                   # We use distance function when training 
    
   # plotNetwork(model)
    
    return model

def SiameseNetworkKNN():
    
    base_model = getModel()

    input_1 = Input(shape=(n_features,), name = 'input_1of2')                   # Only need a single network for applying kNN
    vect_output_1 = base_model(input_1)                                         # Add input layer to model

    model = Model(input_1, vect_output_1)                                       # One input - One output vector (embdedded space)
    
   # plotNetwork(model)
    
    return model
  
def plot_curves(history):
    
    # Lists to store training and validation accuracies for each fold
    train_accuracies = []
    train_losses = []
    val_accuracies = []
    val_losses = []

    # Append training and validation metrics to the lists   
    train_accuracies.append(history.history['accuracy'])
    train_losses.append(history.history['loss'])

    val_accuracies.append(history.history['val_accuracy'])
    val_losses.append(history.history['val_loss'])

    # Get the average training and validation metrics
    avg_train_accuracy = np.mean(train_accuracies, axis=0)
    avg_train_loss = np.mean(train_losses, axis=0)

    avg_val_accuracy = np.mean(val_accuracies, axis=0)
    avg_val_loss = np.mean(val_losses, axis=0)

    # Obtaining min and max metrics for printing
    min_loss_index = np.argmin(avg_val_loss)
    max_accuracy_index = np.argmax(avg_val_accuracy)


    print('min loss: {} at index {}'.format(np.min(avg_val_loss), min_loss_index))
    print('max accuracy: {} at index {}'.format(np.max(avg_val_accuracy), max_accuracy_index))

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 8))

    # Plot training and validation loss
    axs[0].plot(avg_train_loss, label='Training Loss')
    axs[0].plot(avg_val_loss, label='Validation Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Plot training and validation accuracy
    axs[1].plot(avg_train_accuracy, label='Training Accuracy')
    axs[1].plot(avg_val_accuracy, label='Validation Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    # Adjust layout for better spacing
    plt.tight_layout()
    print('plotted')

    # Show the plot
    plt.show()
    
def plot_confusion_matrix(y_true, y_pred, title):
    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels = classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False, 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'{title} Confusion Matrix')
    plt.show()
    

def cosine_annealing(epoch):
    
    T_max = 200
    min_val = 1E-06
    max_val =  0.00604

    return min_val + 0.5 * (max_val - min_val) * (1 + np.cos((epoch*np.pi) / T_max))
      
def train_model():
             
    model = SiameseNetwork()
    
    lr_scheduler = LearningRateScheduler(cosine_annealing)
    
    adamW = AdamW(
        learning_rate = 0.00604,
        weight_decay = 0.0)
    
    model.compile(loss=ContrastiveLoss,
                  optimizer=adamW)
    
    history = model.fit(
        [x1_train, x2_train], y_similar, 
        epochs=200,  batch_size = 30000, callbacks = [lr_scheduler],
        validation_data = ([x1_val[:1000], x2_val[:1000]], y_val[:1000]))
    
    #plot_curves(history)
    
    model.save('test.keras')
   
    return model

def make_predictions():
    
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
    
def make_predictions_KNN():
    
    # Data for fitting KNN is different in its structure
    KNN_x_train = X_train
    KNN_y_train = y_train
    
    KNN_x_val = x1_val[::25] .reset_index(drop=True)                            # Our test sample is in every 25th i of even rows
    KNN_y_val = y[::50].reset_index(drop=True)                                  # Our test sample label is in every 50th i of all rows
    
    model_KNN = SiameseNetworkKNN()
    #model_KNN.load_weights('test.keras')
    
    embeddings_train = model_KNN.predict(KNN_x_train)
      
    knn = KNeighborsClassifier(n_neighbors = len(classes))
    knn.fit(embeddings_train, KNN_y_train)
    
    x_val = x1_val[::25]
    
    embeddings_val = model_KNN.predict(KNN_x_val)
    
    labels = knn.predict(embeddings_val)
    
    return labels
        

def metrics_report(report_dict):
      
    
    precision = [report_dict[class_name]['precision'] for class_name in classes]
    recall = [report_dict[class_name]['recall'] for class_name in classes]
    f1_score = [report_dict[class_name]['f1-score'] for class_name in classes]
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Bar chart for precision
    axs[0].bar(classes, precision, color='blue')
    axs[0].set_title('Precision')
    
    # Bar chart for recall
    axs[1].bar(classes, recall, color='green')
    axs[1].set_title('Recall')
    
    # Bar chart for f1-score
    axs[2].bar(classes, f1_score, color='orange')
    axs[2].set_title('F1-Score')
    
    plt.tight_layout()
    plt.show()
    
    # Rotate x-axis labels for better visibility
    for ax in axs:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Show the plot
    plt.show()

# Set fixed seed for reproducibility
seed_value = 4564
set_rand_seed(seed_value) 

n_features_list = list(range(1,26))


weighted_avgs = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': [],
}

weighted_avgs_KNN = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': [],
}



for n_features in n_features_list: 
    
    scaler = None

    # Select n features - 16 found to do well
    print(n_features)
    selected_features = feature_selection(n_features)
    print(selected_features)
    
    # Get training and validation data with only selected features
    x1_train, x2_train, X_train, y_train,  y_similar, _ = getData('slowloris_complete_200.csv')
    x1_val, x2_val, _, y,  y_val, y_true = getData('Validation_Pairs_Equal.csv')
    
    # Get list of classes 
    classes = y.iloc[1::2].unique()
    
    model = train_model()
    
    # Making predictions - relying on lots of global vars...
    normal_predictions = make_predictions()
    KNN_predictions = make_predictions_KNN()
    
    # Our true labels were..
    y_true = y[::50]
    
    # Accuracy Scores
    accuracy = accuracy_score(y_true, normal_predictions)
    accuracy_KNN = accuracy_score(y_true, KNN_predictions)
    
    print(f'Siamese achieved: {accuracy*100}%')
    print(f'Siamese KNN achieved: {accuracy_KNN*100}%')
    
    # Classification Reports
    report_dict = classification_report(y_true, normal_predictions,
                                               output_dict=True)
    
    report_dict_KNN = classification_report(y_true, KNN_predictions,
                                               output_dict=True)
    
    weighted_avgs['accuracy'].append(accuracy)
    weighted_avgs['precision'].append(report_dict['weighted avg']['precision'])
    weighted_avgs['recall'].append(report_dict['weighted avg']['recall'])
    weighted_avgs['f1'].append(report_dict['weighted avg']['f1-score'])
    
    weighted_avgs_KNN['accuracy'].append(accuracy_KNN)
    weighted_avgs_KNN['precision'].append(report_dict_KNN['weighted avg']['precision'])
    weighted_avgs_KNN['recall'].append(report_dict_KNN['weighted avg']['recall'])
    weighted_avgs_KNN['f1'].append(report_dict_KNN['weighted avg']['f1-score'])
    
    # Plot confusion matrices
    
    #plot_confusion_matrix(y_true, normal_predictions, 'Without KNN')
    #plot_confusion_matrix(y_true, KNN_predictions, 'With KNN')





