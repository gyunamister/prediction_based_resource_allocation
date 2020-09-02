# Experiment on an artificial event log - RQ1, RQ2 with proposed method

# Threshold for the prediction accuracy
threshold=(0.3)

# Each file contains different number of instances for resource allocation
# test_path_array=("./sample_data/artificial/testlog_0806_1_40.csv" "./sample_data/artificial/testlog_0806_1_60.csv" "./sample_data/artificial/testlog_0806_1_80.csv" "./sample_data/artificial/testlog_0806_1_100.csv" "./sample_data/artificial/testlog_0806_1_120.csv" "./sample_data/artificial/testlog_0806_1_140.csv")
test_path_array=("./sample_data/artificial/testlog_0806_1_90.csv")

for alpha in ${threshold[@]}; do
	for test_path in ${test_path_array[@]}; do
		python suggested_main.py --alpha $alpha --beta $alpha --test_path $test_path --exp_name 'exp_1'
	done;
done;