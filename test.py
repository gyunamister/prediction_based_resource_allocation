import pandas as pd;

def read_act_res_mat(path="./sample_data/new_resource_0806_1.csv"):
	"""Read activity-resource matrix which specifies the processing time

	Keyword arguments:
	path -- file path
	"""
	act_res_mat = pd.read_csv(path)
	act_res_mat['Resource'] = 'Resource'+act_res_mat['Resource'].astype('str')
	act_res_mat = act_res_mat.set_index('Resource')
	act_res_mat = act_res_mat.to_dict()
	return act_res_mat


if __name__ == '__main__':
	act_res_mat = read_act_res_mat('./sample_data/real/modi_BPI_2012_dropna_filter_act.csv');
	print(act_res_mat)