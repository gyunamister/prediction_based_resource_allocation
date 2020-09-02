#Experiment on an artificial event log - RQ1, RQ2 with baseline approach

#test_path_array=("./sample_data/artificial/testlog_0806_1_120.csv" )
# test_path_array=("./sample_data/artificial/testlog_0806_1_40.csv" "./sample_data/artificial/testlog_0806_1_60.csv" "./sample_data/artificial/testlog_0806_1_80.csv" "./sample_data/artificial/testlog_0806_1_100.csv" "./sample_data/artificial/testlog_0806_1_120.csv" "./sample_data/artificial/testlog_0806_1_140.csv")
test_path_array=("./sample_data/artificial/testlog_0806_1_90.csv")


for test_path in ${test_path_array[@]}; do
	python baseline_main.py --mode 'test' --test_path $test_path --exp_name 'exp_1-baseline'
done;
