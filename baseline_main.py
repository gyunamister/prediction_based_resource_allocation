from optimizer.baseline import BaseOptimizer

import argparse

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', default='test', type=str)
	parser.add_argument('--test_path', default='./result/testlog_0121_1_100.csv', type=str)
	parser.add_argument('--date', default='2012-03-01', type=str)
	parser.add_argument('--exp_name', default='exp', type=str)
	args = parser.parse_args()

	Opt = BaseOptimizer()

	if args.mode == 'test':
		"""Experiment on an artificial event log"""
		res_info_path = "./sample_data/artificial/new_resource_0806_1.csv"
		Opt.main(test_path=args.test_path, mode=args.mode, res_info_path=res_info_path, date=args.date, exp_name=args.exp_name)

	else:
		"""Experiment on an real-life event log"""
		org_log_path = './sample_data/real/modi_BPI_2012_dropna_filter_act.csv'
		Opt.main(org_log_path = org_log_path, test_path = args.test_path, mode=args.mode, date=args.date, exp_name=args.exp_name)