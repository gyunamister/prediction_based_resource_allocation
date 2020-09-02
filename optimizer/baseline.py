import sys
import os
from pathlib import Path
import networkx as nx
import time
import numpy as np
from collections import OrderedDict
import pickle
import pandas as pd
import random
import copy

p = Path(__file__).resolve().parents[2]
sys.path.append(os.path.abspath(str(p)))

from PyProM.src.data.Eventlog import Eventlog
from object.object import Instance, Resource

from prediction.model import net


class BaseOptimizer(object):
	def __init__(self, *args, **kwargs):
		super(BaseOptimizer, self).__init__(*args, **kwargs)
		self.w_comp_time = list()

	def read_act_res_mat(self, path="./sample_data/new_resource_0806_1.csv"):
		"""Read activity-resource matrix which specifies the processing time

		Keyword arguments:
		path -- file path
		"""
		act_res_mat = pd.read_csv(path)
		act_res_mat['Resource'] = 'Resource'+act_res_mat['Resource'].astype('str')
		act_res_mat = act_res_mat.set_index('Resource')
		act_res_mat = act_res_mat.to_dict()
		return act_res_mat

	def load_data(self,path):
		"""Load eventlog

		Keyword arguments:
		path -- file path
		"""
		eventlog = Eventlog.from_txt(path, sep=',')
		eventlog = eventlog.assign_caseid('CASE_ID')
		eventlog = eventlog.assign_activity('Activity')
		eventlog = eventlog.assign_resource('Resource')
		self.activities = list(set(eventlog['Activity']))
		return eventlog

	def load_real_data(self,path):
		"""Load real-life log (Requires modification according to the schema)

		Keyword arguments:
		path -- file path
		"""
		eventlog = Eventlog.from_txt(path, sep=',')
		eventlog = eventlog.assign_caseid('CASE_ID')
		eventlog = eventlog.assign_activity('Activity')
		eventlog['Resource'] = eventlog['Resource'].astype(int)
		eventlog = eventlog.assign_resource('Resource')
		eventlog = eventlog.assign_timestamp(name='StartTimestamp', new_name='StartTimestamp', _format = '%Y.%m.%d %H:%M:%S', errors='raise')

		def to_minute(x):
			t = x.time()
			minutes = t.hour * 60 + t.minute
			return minutes

		eventlog['Start'] = eventlog['StartTimestamp'].apply(to_minute)
		return eventlog

	def initialize_real_instance(self, eventlog):
		"""Initialize real instance
		Difference between test and real instance
		1. Real - using date info.
		2. Real - release time is set to the appearing time of an instance

		Keyword arguments:
		eventlog -- test log
		"""
		instance_set = list()
		activity_trace = eventlog.get_event_trace(workers=4, value='Activity')
		resource_trace = eventlog.get_event_trace(4,'Resource')
		#eventlog['StartDate'] = pd.to_datetime(eventlog["StartDate"], format='%Y.%m.%d')
		date_trace = eventlog.get_event_trace(workers=4, value='StartDate')
		time_trace = eventlog.get_event_trace(workers=4, value='Start')
		dur_trace = eventlog.get_event_trace(workers=4, value='Duration')
		weight_trace = eventlog.get_event_trace(workers=4, value='weight')
		#target_date=pd.to_datetime(self.date)

		for case in date_trace:
			for j, time in enumerate(date_trace[case]):
				if time >= self.date:
					initial_index =j-1
					release_time = time_trace[case][j]
					break
			weight = min(weight_trace[case])
			instance = Instance(name=case, weight=weight, release_time=release_time, act_sequence=activity_trace[case], res_sequence=resource_trace[case],dur_sequence=dur_trace[case], initial_index=initial_index)
			instance_set.append(instance)

		return instance_set

	def initialize_test_instance(self, eventlog):
		"""Initialize test instance

		Keyword arguments:
		eventlog -- test log
		"""
		instance_set = list()
		activity_trace = eventlog.get_event_trace(workers=4, value='Activity')
		resource_trace = eventlog.get_event_trace(4,'Resource')
		time_trace = eventlog.get_event_trace(workers=4, value='Start')
		dur_trace = eventlog.get_event_trace(workers=4, value='Duration')
		weight_trace = eventlog.get_event_trace(workers=4, value='weight')
		for case in activity_trace:
			release_time = min(time_trace[case])
			#release_time = 0
			weight = min(weight_trace[case])
			instance = Instance(name=case, weight=weight, release_time=release_time, act_sequence=activity_trace[case], res_sequence=resource_trace[case],dur_sequence=dur_trace[case])
			instance_set.append(instance)
		return instance_set

	def initialize_real_resource(self, test_log):
		"""Initialize real instance
		No difference at the moment

		Keyword arguments:
		test_log -- test log
		"""
		resource_set = list()
		resource_list = sorted(list(test_log.get_resources()))
		for res in resource_list:
			act_list = list(test_log.loc[test_log['Resource']==res,'Activity'].unique())
			resource = Resource(res, act_list)
			resource_set.append(resource)
		return resource_set

	def initialize_test_resource(self, eventlog):
		"""Initialize test resource

		Keyword arguments:
		eventlog -- test log
		"""
		resource_set = list()
		resource_list = sorted(list(eventlog.get_resources()))
		for res in resource_list:
			act_list = list(eventlog.loc[eventlog['Resource']==res,'Activity'].unique())
			resource = Resource(res, act_list)
			resource_set.append(resource)
		return resource_set

	def set_basic_info(self, eventlog):
		"""set basic info. for instances

		Keyword arguments:
		eventlog -- test log
		"""

		# To be aligned with the entire log, we load the information generated from entire log
		if self.mode == 'test':
			with open('./prediction/checkpoints/traininglog_0806_1.csv_activities.pkl', 'rb') as f:
				activities = pickle.load(f)
			with open('./prediction/checkpoints/traininglog_0806_1.csv_resources.pkl', 'rb') as f:
				resources = pickle.load(f)
		else:
			with open('./prediction/checkpoints/modi_BPI_2012_dropna_filter_act.csv_activities.pkl', 'rb') as f:
				activities = pickle.load(f)
			with open('./prediction/checkpoints/modi_BPI_2012_dropna_filter_act.csv_resources.pkl', 'rb') as f:
				resources = pickle.load(f)
		act_char_to_int = dict((str(c), i) for i, c in enumerate(activities))
		act_int_to_char = dict((i, str(c)) for i, c in enumerate(activities))
		res_char_to_int = dict((str(c), i) for i, c in enumerate(resources))
		res_int_to_char = dict((i, str(c)) for i, c in enumerate(resources))

		# for contextual information
		self.queue = OrderedDict()
		for act in activities:
			if act != '!':
				self.queue[act] = 0

		# maxlen information
		activity_trace = eventlog.get_event_trace(4,'Activity')
		trace_len = [len(x) for x in activity_trace.values()]
		maxlen = max(trace_len)

		# set info.
		Instance.set_activity_list(activities)
		Instance.set_resource_list(resources)
		Instance.set_act_char_to_int(act_char_to_int)
		Instance.set_act_int_to_char(act_int_to_char)
		Instance.set_res_char_to_int(res_char_to_int)
		Instance.set_res_int_to_char(res_int_to_char)
		Instance.set_maxlen(maxlen)

	def load_model(self, checkpoint_dir, model_name):
		"""load prediction model

		Keyword arguments:
		checkpoint_dir -- directory path
		model_name -- decide which model to load
		"""
		model = net()
		model.load(checkpoint_dir, model_name)
		return model

	def prepare_test(self, test_path, res_info_path):
		"""prepare experiment on the artificial log

		Keyword arguments:
		test_path -- path to the test log
		res_info_path -- path to the activity-resource processing time
		"""

		checkpoint_dir = './prediction/checkpoints/'
		modelname_next_act = 'traininglog_0806_1.csv' + 'next_activity'
		modelname_next_time = 'traininglog_0806_1.csv' + 'next_timestamp'

		# load prediction model
		model_next_act = self.load_model(checkpoint_dir, modelname_next_act)
		model_next_time = self.load_model(checkpoint_dir, modelname_next_time)

		# set prediction model
		Instance.set_model_next_act(model_next_act)
		Instance.set_model_next_time(model_next_time)

		# load log
		test_log = self.load_data(path=test_path)

		#initialize resource set
		resource_set = self.initialize_test_resource(test_log)

		#create act-res matrix
		self.act_res_mat = self.read_act_res_mat(res_info_path)

		# initialize instance set
		instance_set = self.initialize_test_instance(test_log)

		#Set attributes of instance -> to be used to gernerate input for prediction
		self.set_basic_info(test_log)

		return resource_set, instance_set

	def prepare_real(self, test_path, org_log_path):
		"""prepare experiment on the real log

		Keyword arguments:
		test_path -- path to the test log
		org_log_path -- path to the entire log
		"""

		checkpoint_dir = './prediction/checkpoints/'
		modelname_next_act = 'modi_BPI_2012_dropna_filter_act.csv' + 'next_activity'
		modelname_next_time = 'modi_BPI_2012_dropna_filter_act.csv' + 'next_timestamp'

		# load prediction model
		model_next_act = self.load_model(checkpoint_dir, modelname_next_act)
		model_next_time = self.load_model(checkpoint_dir, modelname_next_time)

		# set prediction model
		Instance.set_model_next_act(model_next_act)
		Instance.set_model_next_time(model_next_time)

		# load eventlog
		eventlog = self.load_real_data(path=org_log_path)

		# load test log
		test_log = self.load_real_data(path=test_path)
		self.num_cases = len(set(test_log['CASE_ID']))
		self.avg_weight = test_log['weight'].mean()

		#no act-res matrix
		self.act_res_mat = None

		# initialize instance set
		instance_set = self.initialize_real_instance(test_log)

		#initialize resource set
		resource_set = self.initialize_real_resource(test_log)

		#Set attributes of instance -> to be used to gernerate input for prediction
		self.set_basic_info(eventlog)

		return resource_set, instance_set


	def update_ongoing_instances(self, instance_set, ongoing_instance, t):
		"""include released instances to the ongoing instance set

		Keyword arguments:
		instance_set -- all instances for resource allocation
		ongoing_instance -- ongoing instance set
		t -- current time
		"""
		for i in instance_set:
			if i.get_release_time() == t:
				ongoing_instance.append(i)
		return ongoing_instance

	def update_object(self, ongoing_instance, resource_set, t):
		"""create the bipartite graph with the prediction results

		Keyword arguments:
		ongoing_instance -- ongoing instance set
		resource_set -- all resources for resource allocation
		t -- current time
		"""
		# if resource is free, set the status to 'True'
		for j in resource_set:
			if j.get_next_actual_ts() == t:
				j.set_status(True)

		# if resource is free, set the status to 'True'
		for i in ongoing_instance:
			if i.get_next_actual_ts() == t:
				cur_actual_act = i.get_cur_actual_act()
				if cur_actual_act != False:
					self.queue[cur_actual_act] -= 1
				i.set_status(True)
			elif i.get_next_actual_ts() < t:
				i.update_weight()

		# generate bipartite graph
		ready_instance = [x for x in ongoing_instance if x.get_status()==True]
		ready_resource = [x for x in resource_set if x.get_status()==True]
		G = nx.DiGraph()
		for i in ready_instance:
			actual_act = i.get_next_actual_act()
			for j in ready_resource:
				if actual_act in j.get_skills():
					G.add_edge('s',i, capacity=1)
					G.add_edge(j,'t',capacity=1)
					weight = i.get_weight()
					cost = weight * (-1)
					G.add_edge(i,j,weight=cost,capacity=1)
		return G


	def update_plan(self, G, t):
		"""solve the min-cost max-flow algorithm to find an optimal schedule

		Keyword arguments:
		G -- bipartite graph
		t -- current time
		"""
		nodes=G.nodes()
		if len(nodes)!=0:
			M = nx.max_flow_min_cost(G, 's', 't')
		else:
			M=False
		return M


	def execute_plan(self, ongoing_instance, resource_set, M, t):
		"""execute the resource allocation and update the situation accordingly.

		Keyword arguments:
		ongoing_instance -- ongoing instance set
		resource_set -- all resources for resource allocation
		M -- optimal schedule
		t -- current time
		"""

		ready_instance = [x for x in ongoing_instance if x.get_status()==True]
		ready_resource = [x for x in resource_set if x.get_status()==True]
		if M!=False:
			for i in M:
				if i in ready_instance:
					for j, val in M[i].items():
						# check if there is a flow
						if val==1 and j in ready_resource:
							cur_pred_dur, cur_time_uncertainty = i.predict_next_time(self.queue, context=True, pred_act=i.get_next_actual_act(), resource=j.get_name())
							i.set_pred_act_dur(j, cur_pred_dur, cur_time_uncertainty)
							i.update_actuals(t, j, self.mode, self.act_res_mat,self.queue)

							j.set_next_pred_ts(i.get_next_pred_ts())
							j.set_next_ts_uncertainty(i.get_next_ts_uncertainty(j))
							j.set_next_actual_ts(i.get_next_actual_ts())
							j.set_status(False)

							# update contextual information
							cur_actual_act = i.get_cur_actual_act()
							if cur_actual_act != False:
								self.queue[cur_actual_act] += 1

							i.clear_pred_act_dur()
							# to implement FIFO rule
							i.reset_weight()


	def update_completes(self, completes, ongoing_instance, t):
		"""check if instance finishes its operation

		Keyword arguments:
		completes -- set of complete instances
		ongoing_instance -- ongoing instance set
		t -- current time
		"""
		for i in ongoing_instance:
			finished = i.check_finished(t)
			if finished==True:
				cur_actual_act = i.get_cur_actual_act()
				self.queue[cur_actual_act] -= 1
				i.set_weighted_comp()
				ongoing_instance.remove(i)
				completes.append(i)
				self.w_comp_time.append(i.get_weighted_comp())
				"""
				with open("./exp_result/exp_7.txt", "a") as f:
					f.write("{}-{}: start at {}, end at {}, weighted_comp = {} \n".format(i.get_name(), i.get_weight(), i.release_time, i.get_next_actual_ts(), i.get_weighted_comp()))
				"""
		return completes

	def main(self, test_path, mode, date, exp_name, **kwargs):
		time1 = time.time()
		t=0
		#initialize
		ongoing_instance = list()
		completes = list()
		self.mode = mode
		self.date = date

		if mode=='test':
			if "res_info_path" in kwargs:
				res_info_path = kwargs['res_info_path']
			else:
				raise AttributeError("Resource Information is required")
			resource_set, instance_set = self.prepare_test(test_path, res_info_path)

		elif mode == 'real':
			if 'org_log_path' in kwargs:
				org_log_path = kwargs['org_log_path']
			else:
				raise AttributeError("no org_log_path given.")
			resource_set, instance_set = self.prepare_real(test_path, org_log_path )
			print("num resource:{}".format(len(resource_set)))

		else:
			raise AttributeError('Optimization mode should be given.')


		while len(instance_set) != len(completes):
			print("{} begins".format(t))
			#ongoing instance를 추가
			ongoing_instance = self.update_ongoing_instances(instance_set, ongoing_instance, t)
			#print('current ongoing instance: {}'.format(len(ongoing_instance)))
			G = self.update_object(ongoing_instance, resource_set,t)
			#print('current cand instance and resource: {}, {}'.format(cand_instance, cand_resource))
			M = self.update_plan(G,t)
			#print('current matching: {}'.format(M))
			self.execute_plan(ongoing_instance, resource_set, M, t)
			completes = self.update_completes(completes, ongoing_instance, t)
			print('current completes: {}'.format(len(completes)))
			t+=1
		time2 = time.time()

		total_weighted_sum = sum(self.w_comp_time)
		total_computation_time = (time2-time1)

		print("total weighted sum: {}".format(total_weighted_sum))
		print('suggested algorithm took {:.1f} s'.format(total_computation_time))
		if self.mode=='real':
			with open("./exp_result/{}.txt".format(exp_name), "a") as f:
				f.write("Baseline: {}, num_cases: {}, avg_weight: {} \n {}, {} \n".format(test_path, self.num_cases, self.avg_weight, total_weighted_sum, total_computation_time))
		else:
			with open("./exp_result/{}.txt".format(exp_name), "a") as f:
				f.write("Baseline: {} \n {}, {} \n".format(test_path, total_weighted_sum, total_computation_time))
