## AI (Machine Learning) for Cybersecurity using Limited Data ##

Author: Christopher Abi-aad
Supervised by Robert Atkinson

The proposed model incorporates a Siamese Network which leverages transfer learning.
Model is trained with no constraint on number of samples but initially excludes an attack class
It is then retrained with a limited number of samples from the excluded class to simulate a real world scenario

Please see @@ Findings folder for results

-> ## Data
	Folder containing csv files for samples of each attack class with final feature set

-> ## Limited Data
	Folder containing subfolder for each class
	Each subfolder contains the base csv file for that attack (which has pairs that excludes that class)
	Also contains csv files which have pairs with varying numbers of samples for that class, denoted by '_N.csv'
	Most of these csv files are hidden in the 'hide' folder as they are not required for the final script

-> @@ Findings
	Contains its own readme
	Includes summary of findings including CM, Hyperparameters, and trained model's weights

-> Pre Processing
	Contains its own read me
	Includes script to process the raw CICIDS2017 dataset

-> Archive
	Contains its own read me
	Prev files used in the project which may be useful for reproduction of FNN

-> random_pair_sampling_TL-FSL.py:
	A script used to generate both training and validation pairs for a Siamese Network
	The script relies on the csv files contained within ## Data - each csv file is treated as a class
	getTrainingPairs(n_pairs) returns an equal number of similar and dissimilar pairs - equal across all classes
	getValidationPairs(n_test_samples_per_class) returns validation set with N samples per class

-> Siamese_Network_TL-FSL.py
	The main script used to produce the reported findings
	Relies on csvs in ## Limited Data & Validation_Pairs_x.csv
	Contains a siamese network which leverages transfer learning
	Specify limited class & use specified seed for reproduction
	In its current form, the script loads the weights from trained models defined within ## Limited Data
	However, hyperparamter optimisation section can be uncommented to retrain models on the fly - see line 468 & 505


-> Validation_pairs.csv
	Validation_pairs_attack.csv: Pairs which do not include the specified attack class used to validation during initial training
	Validation_pairs_equal.csv: Pairs with all classes used to validate retrained model
