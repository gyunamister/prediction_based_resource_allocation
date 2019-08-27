from optimizer.suggested import SuggestedOptimizer

import argparse

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', default='test', type=str)
	parser.add_argument('--alpha', default=0.2, type=float)
	parser.add_argument('--beta', default=0.2, type=float)
	parser.add_argument('--test_path', default='./result/modi_BPI_2012_2012.3.1.csv', type=str)
	parser.add_argument('--date', default='2012-03-01', type=str)
	parser.add_argument('--precision', default=0.0, type=float)
	parser.add_argument('--exp_name', default='exp', type=str)
	args = parser.parse_args()

	if args.mode == 'test':
		res_info_path = "./sample_data/artificial/new_resource_0806_1.csv"
		Opt = SuggestedOptimizer()
		Opt.main(test_path=args.test_path, mode=args.mode, alpha=args.alpha, beta=args.beta, res_info_path=res_info_path, precision=args.precision, date=args.date, exp_name=args.exp_name)

	else:
		#real
		Opt = SuggestedOptimizer()
		org_log_path = './sample_data/real/modi_BPI_2012_dropna_filter_act.csv'
		Opt.main(org_log_path = org_log_path, test_path = args.test_path, mode=args.mode, alpha=args.alpha, beta=args.beta, precision=args.precision, date=args.date, exp_name=args.exp_name)
