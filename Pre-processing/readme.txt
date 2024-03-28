## Pre-processing ##

Author: Christopher Abi-aad 

-> The folder contains 5 individual scripts transform raw .pcap files to .csv files with flow-level features, segregated by class.
-> Scripts are included seperately due to the insufficient memory using 8 GB RAM + long run times
-> The scripts should be run in order from 01 to 05
-> Each script should be in a cwd where 5 folders exist:
	'Monday-WorkingHours' : includes 'Monday-WorkingHours.pcap'
	..
	..
	'Friday-WorkingHours' : includes 'Friday-WorkingHours.pcap'


====================================================================================================================================
01_parse_pcap.py

-> Input: 'Monday-WorkingHours.pcap'
-> Output: 'Monday-WorkingHours.csv' - csv file containing header information at packet level

====================================================================================================================================
02_group_flows.py

-> Input: 'Monday-WorkingHours.csv' - 
-> Output: 'Monday-WorkingHours_sorted.csv' - csv file with packets grouped by flow

====================================================================================================================================
03_split_parsed_csv.py

-> Input: 'Monday-WorkingHours_sorted.csv' 
-> Output: 'Monday-WorkingHours_0.csv'
	    ..
 	    ..
	    'Monday-WorkingHours_n.csv'  - Splits the csv files to more manageable pieces while maintaining flows
====================================================================================================================================
04_obtain_flow_features.py

-> Input: 'Monday-WorkingHours_0.csv'
	    ..
 	    ..
	    'Monday-WorkingHours_n.csv'

-> Output: 'Monday-WorkingHours_0_flows.csv'
	    ..
 	    ..
	    'Monday-WorkingHours_n_flows.csv' - Calculates flow level features in fwd and bwd direction

====================================================================================================================================
05_split_by_label.py


-> Input: 'Monday-WorkingHours_0_flows.csv'
	    ..
 	    ..
	    'Monday-WorkingHours_n_flows.csv'

-> Output: 'slowloris.csv'		- Individual files for each attack 
	   'hulk.csv'			- Packet level features have been dropped to reduce data size (easily changed if needed)
	    ..				- Duplicates are also dropped
	    ..


