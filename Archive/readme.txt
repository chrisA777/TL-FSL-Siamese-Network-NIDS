## Archive ##

-> Some archived files which may be useful for:
	-> Reproduction of baseline model
	-> Generating training pairs for model without transfer learning
	-> Evaluating feature set on Siamese Network


-> create_dataset.py: creates a subset of the original dataset w/ equal class distribution for FNN
-> FNN.py: requires a csv file from create_dataset.py
	   trains and evaluates FNN with a number of plots
-> random_pair_sampling_v2.py: a prev version of pair sampling script which was used to generate pairs without excluded a class
-> Siamese_Network_FSL_Feature_Evaluation.py: plots model performance against number of features