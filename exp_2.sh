#Experiment on an artificial event log - RQ3 with proposed method

# Threshold for the prediction accuracy
threshold=(0.2 )
# Each file contains different number of instnaces for resource allocation
test_path_array=("./sample_data/artificial/testlog_0806_1_90.csv" "./sample_data/artificial/testlog_0806_1_110.csv" "./sample_data/artificial/testlog_0806_1_130.csv" "./sample_data/artificial/testlog_0806_1_150.csv")
precision_array=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)


for alpha in ${threshold[@]}; do
	for test_path in ${test_path_array[@]}; do
		for precision in ${precision_array[@]}; do
			python suggested_main.py --alpha $alpha --beta $alpha --test_path $test_path --precision $precision --exp_name 'exp_2'
		done;
	done;
done;