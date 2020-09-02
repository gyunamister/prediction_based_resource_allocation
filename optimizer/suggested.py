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

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))

        return ret
    return wrap

class SuggestedOptimizer(object):
	def __init__(self, *args, **kwargs):
		super(SuggestedOptimizer, self).__init__(*args, **kwargs)
		self.w_comp_time = list()
		self.pred_time = list()
		self.act_res_mat = None

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
			weight = min(weight_trace[case])
			instance = Instance(name=case, weight=weight, release_time=release_time, act_sequence=activity_trace[case], res_sequence=resource_trace[case],dur_sequence=dur_trace[case])
			instance_set.append(instance)
		return instance_set

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
		date_trace = eventlog.get_event_trace(workers=4, value='StartDate')
		time_trace = eventlog.get_event_trace(workers=4, value='Start')
		dur_trace = eventlog.get_event_trace(workers=4, value='Duration')
		weight_trace = eventlog.get_event_trace(workers=4, value='weight')

		for case in date_trace:
			for j, time in enumerate(date_trace[case]):
				if time == self.date:
					initial_index =j-1
					release_time = time_trace[case][j]
					break
			weight = min(weight_trace[case])
			instance = Instance(name=case, weight=weight, release_time=release_time, act_sequence=activity_trace[case], res_sequence=resource_trace[case],dur_sequence=dur_trace[case], initial_index=initial_index)
			instance_set.append(instance)

		return instance_set

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

		# (CHANGED)
		est_dir = './prediction/estimation/'
		estname_next_time = 'modi_BPI_2012_dropna_filter_act.csv' + 'next_timestamp'
		# load estimation model
		est_next_time = self.load_model(est_dir, estname_next_time)

		# set prediction model
		Instance.set_est_next_time(est_next_time)

		# load eventlog
		eventlog = self.load_real_data(path=org_log_path)

		# load test log
		test_log = self.load_real_data(path=test_path)

		#no act-res matrix
		self.act_res_mat = None

		# initialize instance set
		instance_set = self.initialize_real_instance(test_log)

		#initialize resource set
		resource_set = self.initialize_real_resource(test_log)

		#Set attributes of instance -> to be used to gernerate input for prediction
		self.set_basic_info(eventlog)

		return resource_set, instance_set

	#@timing
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

	#@timing
	def update_object(self, ongoing_instance, resource_set, t):
		"""create the bipartite graph with the prediction results

		Keyword arguments:
		ongoing_instance -- ongoing instance set
		resource_set -- all resources for resource allocation
		t -- current time
		"""
		G = nx.DiGraph()
		# if resource is free, set the status to 'True'
		for j in resource_set:
			if j.get_next_actual_ts() <= t:
				j.set_status(True)

		# if instance is free, set the status to 'True'
		"""
		for i in ongoing_instance:
			if i.get_next_actual_ts() <= t:
				i.set_status(True)
		"""

		# if resource is free, set the status to 'True'
		for i in ongoing_instance:
			# if instance finishes current operation,
			if i.get_next_actual_ts() == t:
				# set the status to 'True'
				i.set_status(True)

				# update contextual information
				cur_actual_act = i.get_cur_actual_act()
				if cur_actual_act != False:
					self.queue[cur_actual_act] -= 1

				if self.exp_name != 'exp_2':
					# if it has just been released or the next act. prediction was wrong, update the processing time prediction
					if i.first or i.get_next_actual_act() != i.get_next_pred_act():
						i.clear_pred_act_dur()
						for j in resource_set:
							if i.get_next_actual_act() in j.get_skills():
								next_pred_dur, next_time_uncertainty = i.predict_next_time(self.queue, context=True, pred_act=i.get_next_actual_act(), resource=j.get_name())
								# set prediction uncertainty to 0 since it is ready for the next act.
								i.set_next_act_uncertainty(0)
								i.set_pred_act_dur(j, next_pred_dur, 0)
					else:
						#set prediction uncertainty to 0 since it is ready for the next act.
						i.set_next_act_uncertainty(0)
				else:
					# if it has just been released or the next act. prediction was wrong, update the processing time prediction
					if i.first or i.get_next_actual_act() != i.get_next_pred_act():
						i.clear_pred_act_dur()
						for j in resource_set:
							if i.get_next_actual_act() in j.get_skills():
								next_pred_dur, next_time_uncertainty = int(self.act_res_mat[i.get_next_actual_act()][j.get_name()]), 0
								# give noise
								if np.random.uniform(0,1) < 0.5:
									next_pred_dur += self.precision * next_pred_dur
								else:
									next_pred_dur -= self.precision * next_pred_dur
								next_pred_dur = round(next_pred_dur)
								if next_pred_dur == 0:
									next_pred_dur = 1

								i.set_next_act_uncertainty(0)
								i.set_pred_act_dur(j, next_pred_dur, 0)
					else:
						#set prediction uncertainty to 0 since it is ready for the next act.
						i.set_next_act_uncertainty(0)

			# if instance is under operation and the next act. prediction uncertainty is above the threshold, we do not allocate resources for it
			elif i.get_next_actual_ts() > t:
				if i.get_next_act_uncertainty() > self.act_uncertainty:
					continue

			for j in i.get_pred_act_dur_dict().keys():
				# if the processing time prediction uncertainty is above the threshold, we do not include the edge.
				if i.get_next_actual_ts() > t:
					if i.get_next_ts_uncertainty(j) > self.ts_uncertainty and j.get_next_ts_uncertainty() > self.ts_uncertainty:
						continue
				# generate bipartite graph
				G.add_edge('s',i, capacity=1)
				G.add_edge(j,'t',capacity=1)
				weight = i.get_weight()
				pred_dur = i.get_pred_act_dur(j)
				pred_dur += max([i.get_next_pred_ts()-t, j.get_next_pred_ts()-t, 0])
				cost = int(pred_dur / weight * 10)
				G.add_edge(i,j,weight=cost,capacity=1, pred_dur=pred_dur)

		return G

	#@timing
	def update_plan(self, G,t):
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
		#M = MinCost_MaxFlow(s,t) # dict of dict form
		return M

	def modify_plan(self, G, M, t):
		"""if some instances can be handled within the waiting time for best-matched instance, handle the instance who has the maximum weight.
		(We don't use it at the moment)

		Keyword arguments:
		G -- bipartite graph
		t -- current time
		"""
		if M!=False:
			for i, _ in M.items():
				if isinstance(i, Instance)==False:
					continue
				# if some instances can be handled within the waiting time for best-matched instance, handle the instance who has the maximum weight.
				temp_dict = dict()
				for j, val in M[i].items():
					if val==1:
						remaining = i.get_next_actual_ts()-t
						if remaining <= 0:
							break
						in_edges_to_j = G.in_edges([j], data=True)
						for source, dest, data in in_edges_to_j:
							if source.get_status()==True:
								if data['pred_dur'] <= remaining:
									#Also, we should check whether source is already assigned.
									assigned = False
									for r, val in M[source].items():
										if val == 1:
											assigned = True
									if assigned == False:
										temp_dict[source] = source.get_weight()

				if len(temp_dict)!=0:
					new_instance = max(temp_dict, key=temp_dict.get)
					M[i][j] = 0
					M[new_instance][j] = 1
					print("Match changed: from {} to {}, {}".format(i,new_instance, j.get_name()))
		return M


	#@timing
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
						if val==1 and M[j]['t']==1:
							if j in ready_resource:
								# if both instance and resource are ready for resource allocation,
								# update the current situation regarding the instance
								# (CHANGED)
								i.update_actuals(t, j, self.mode, self.act_res_mat,self.queue)

								# update the info. for the resource
								j.set_next_pred_ts(i.get_next_pred_ts())
								j.set_next_ts_uncertainty(i.get_next_ts_uncertainty(j))
								j.set_next_actual_ts(i.get_next_actual_ts())
								j.set_status(False)

								# update contextual information
								cur_actual_act = i.get_cur_actual_act()
								if cur_actual_act != False:
									self.queue[cur_actual_act] += 1

								if self.exp_name != 'exp_2':
									next_pred_act, next_act_uncertainty = i.predict_next_act(self.queue, context=True)
									i.set_next_pred_act(next_pred_act)
									i.set_next_act_uncertainty(next_act_uncertainty)
								else:

									if np.random.uniform(0,1) > self.precision:
										next_pred_act, next_act_uncertainty = i.get_next_actual_act(), 0
									else:
										activities = copy.deepcopy(self.activities)
										activities.remove(i.get_next_actual_act())
										next_pred_act, next_act_uncertainty = random.choice(activities), 0

									i.set_next_pred_act(next_pred_act)
									i.set_next_act_uncertainty(next_act_uncertainty)

								# clear dict for processing time and predict the processing time for the next activity
								i.clear_pred_act_dur()
								for k in resource_set:
									if next_pred_act in k.get_skills():
										if self.exp_name != 'exp_2':
											next_pred_dur, next_time_uncertainty = i.predict_next_time(self.queue, context=True, pred_act=next_pred_act, resource=k.get_name())
										else:
											# give noise
											next_pred_dur, next_time_uncertainty = int(self.act_res_mat[next_pred_act][k.get_name()]), 0
											if np.random.uniform(0,1) < 0.5:
												next_pred_dur += self.precision * next_pred_dur
											else:
												next_pred_dur -= self.precision * next_pred_dur
											next_pred_dur = round(next_pred_dur)
											if next_pred_dur <= 0:
												next_pred_dur = 1
										i.set_pred_act_dur(k, next_pred_dur, next_time_uncertainty)


	#@timing
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
				# update the contextual information
				cur_actual_act = i.get_cur_actual_act()
				self.queue[cur_actual_act] -= 1

				# compute the total weighted completion time and computation time
				i.set_weighted_comp()
				ongoing_instance.remove(i)
				completes.append(i)
				self.w_comp_time.append(i.get_weighted_comp())
				self.pred_time += i.get_pred_time_list()
				"""
				with open("./exp_result/exp_6.txt", "a") as f:
					f.write("{}-{}: start at {}, end at {}, weighted_comp = {} \n".format(i.get_name(), i.get_weight(), i.release_time, i.get_next_actual_ts(), i.get_weighted_comp()))
				"""
		return completes

	def main(self, test_path, mode, alpha, beta, precision, date, exp_name, **kwargs):
		time1 = time.time()
		t=0
		#initialize
		ongoing_instance = list()
		completes = list()
		self.exp_name = exp_name
		self.act_uncertainty=alpha
		self.ts_uncertainty=beta
		self.precision = precision
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
			#Add ongoing instance
			ongoing_instance = self.update_ongoing_instances(instance_set, ongoing_instance, t)
			#print('current ongoing instance: {}'.format(len(ongoing_instance)))

			G = self.update_object(ongoing_instance, resource_set, t)
			#print("{} updated object".format(t))

			M = self.update_plan(G,t)
			#print("{} updated plan".format(t))

			#M = self.modify_plan(G, M,t)
			#print("{} modified plan".format(t))

			self.execute_plan(ongoing_instance, resource_set, M, t)
			#print("{} executed plan".format(t))

			completes = self.update_completes(completes, ongoing_instance, t)

			print('current completes: {}'.format(len(completes)))

			# for log generation
			for i in ongoing_instance:
				cost_dict = dict()
				for j in i.get_pred_act_dur_dict().keys():
					weight = i.get_weight()
					#j.set_duration_dict(i,pred_dur)
					pred_dur = i.get_pred_act_dur(j)
					pred_dur += max([i.get_next_pred_ts()-t, j.get_next_pred_ts()-t, 0])
					cost = int(pred_dur / weight * 10)
					cost_dict[j] = cost
				print("ongoing {} - status: {}, next: {}, cost: {}".format(i.get_name(),i.get_status(), i.get_next_actual_act(),cost_dict))

			t+=1
			if t > 2500:
				print("STOP")
				break
		time2 = time.time()

		total_weighted_sum = sum(self.w_comp_time)
		total_pred_time = sum(self.pred_time)
		total_computation_time = (time2-time1)
		total_opti_time = total_computation_time - total_pred_time

		print("total weighted sum: {}".format(total_weighted_sum))
		print('suggested algorithm took {:.1f} s'.format(total_computation_time))
		print("total time for predictions: {:.1f} s".format(total_pred_time))
		print("total time for optimizations: {:.1f} s".format(total_opti_time))

		with open("./exp_result/{}.txt".format(exp_name), "a") as f:
			f.write("{}, {}, {} \n {}, {}, {}, {}, {}% \n".format(test_path, alpha, beta, total_weighted_sum, total_computation_time, total_pred_time, total_opti_time, self.precision*100))