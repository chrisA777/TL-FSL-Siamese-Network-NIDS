## @@ Findings ##

Author: Christopher Abi-aaad

-> Folder contains 4 subfolders for each attack class:
	-> 'X_initial.csv' : contains the hyperparameters, random seed, and overall accuracy for training without X class
	-> 'X_initial.keras': the weights of the trained model 
	-> 'X_initial.png': confusion matrix for initial training - validated on 'Validation_Pairs_X.csv'
	-> 'X_retrained.csv' : same as before but for retrained model
	-> 'X_retrained.keras': ...
	-> 'X_retrained.png': confusion matrix after retraining - validated on 'Validation_Pairs_Equal.csv'

Note: Load .keras files using:
	
	model = SiameseNetwork()
	model.load_weights('file_name.keras') - Was unable to figure out how to load full model with custom functions so load weights instead