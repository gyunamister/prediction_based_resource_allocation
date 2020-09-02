import sys
import os
import signal
import pandas as pd
import math
import datetime
import numpy as np
from datetime import datetime
from pathlib import Path
from bokeh.plotting import figure, show, output_file
from bokeh.palettes import Inferno256, Viridis256
from bokeh.models import Legend, HoverTool
#to include /Documents like directory
p = Path(__file__).resolve().parents[2]
sys.path.append(os.path.abspath(str(p)))
print(p)
from PyProM.src.data.Eventlog import Eventlog
from PyProM.src.data.xes_reader import XesReader

import collections





class Instance(object):
	def __init__(self, name, release_time, *args, **kwargs):
		super(Instance, self).__init__()
		if 'sequence' in kwargs:
			self.seq = kwargs['sequence']
			self.next_act = self.seq[0]
		elif 'initial_activity' in kwargs:
			self.next_act = kwargs['initial_activity']
		else:
			raise AttributeError('Either sequence or initial_activity should be given.')
		self.name = name
		self.next_ts = release_time
		self.act_index = 0

	def __str__(self):
		return self.name

	def __repr__(self):
		return self.name

	def get_next_act(self):
		return self.next_act

	def get_next_ts(self):
		return self.next_ts

	def set_next_act(self):
		self.act_index += 1
		if self.act_index < len(self.seq):
			"""
			print(self.name)
			print(self.seq)
			print(self.next_ts)
			print(self.next_act)
			print(self.act_index)
			"""
			self.next_act = self.seq[self.act_index]

	def set_next_ts(self, next_ts):
		self.next_ts = next_ts

	def check_finished(self):
		if self.act_index >= len(self.seq):
			return True
		else:
			return False


class Resource(object):
	def __init__(self, name, info, *args, **kwargs):
		super(Resource, self).__init__()
		self.name = name
		self.info = info
		self.next_ts = 0

	def get_skills(self):
		return list(self.info.keys())

	def get_next_ts(self):
		return self.next_ts

	def set_next_ts(self, next_ts):
		self.next_ts = next_ts

	def get_performance(self, act):
		return self.info[act]

class LogGenerator(object):
	def __init__(self, mode, endpoint, res_path, *args, **kwargs):
		self.mode = mode
		self.endpoint = endpoint
		if self.mode == 'test':
			if 'count' in kwargs:
				self.count = kwargs['count']
			interval = round(self.endpoint/(self.count/4)+0.5)
		else:
			interval = 5

		trace_count = self.extract_sequence()
		trace_count = self.modify_sequence(trace_count)
		act_res_mat= self.read_act_res_mat(res_path)

		self.set_release_schedule(trace_count, endpoint, interval)
		self.set_resource_info(act_res_mat)
		self.ongoing_instance = list()
		self.completes = list()
		self.w_instance = list()
		self.w_resource = list()
		self.prod_hbar()
		self.eventlist = list()

	def extract_sequence(self, path="./testlog1_no_noise.csv"):
		eventlog = Eventlog.from_txt(path, sep=',')
		eventlog = eventlog.assign_caseid('Case ID')
		eventlog = eventlog.assign_activity('Activity')
		trace = eventlog.get_event_trace(workers=4, value='Activity')
		trace = trace.values()
		trace = ['_'.join(x) for x in trace]
		trace_count=collections.Counter(trace)
		test_trace_count = trace_count
		return trace_count

	def modify_sequence(self, trace_count):
		if self.mode == 'training':
			"""
			trace_count['Registration_Triage and Assessment_Intravenous_X-ray_Evaluation_Admission_Discharge']+=200
			trace_count['Registration_Triage and Assessment_Intravenous_MRI_Evaluation_Admission_Discharge']-=200
			trace_count['Registration_Triage and Assessment_Blood Test_Diagnosis_Admission_Discharge']+=180
			trace_count['Registration_Triage and Assessment_Blood Test_Urine Test_Diagnosis_Admission_Discharge']-=180
			"""
		elif self.mode == 'test':
			count = int(self.count/4)
			trace_count['Registration_Triage and Assessment_Intravenous_X-ray_Evaluation_Admission_Discharge']=count
			trace_count['Registration_Triage and Assessment_Intravenous_MRI_Evaluation_Admission_Discharge']=count
			trace_count['Registration_Triage and Assessment_Blood Test_Diagnosis_Admission_Discharge']=count
			trace_count['Registration_Triage and Assessment_Blood Test_Urine Test_Diagnosis_Admission_Discharge']=count
		else:
			raise ValueError('mode is not specified')

		"""
		for trace in trace_count:
			trace_count[trace]-=200
		"""
		return trace_count

	def read_act_res_mat(self, path):
		act_res_mat = pd.read_csv(path)
		act_res_mat['Resource'] = 'Resource'+act_res_mat['Resource'].astype('str')
		act_res_mat = act_res_mat.set_index('Resource')
		return act_res_mat

	def set_release_schedule(self, trace_count, endpoint, interval):
		release_schedule = dict()
		tmp=0
		#to produce hbar y_range
		self.total_caseid=list()
		for tc in trace_count:
			count = trace_count[tc]
			sequence = tc.split('_')
			#release_interval = int(endpoint/count) + (interval-int((endpoint/count))%interval)
			release_interval = interval
			print(sequence, release_interval)
			for i in range(1,count+1,1):
				time = release_interval*i
				instance = Instance(name="Case{}".format(tmp+i), release_time=time, sequence=sequence)
				if time not in release_schedule:
					release_schedule[time] = [instance]
				else:
					release_schedule[time].append(instance)
				self.total_caseid.append(instance.name)
			tmp+=count
		self.total_ongoing_instance = tmp
		self.release_schedule = release_schedule

	def set_resource_info(self, act_res_mat):
		resource_set=list()
		for index, row in act_res_mat.iterrows():
			resource_name = index
			resource_info = dict()
			for activity, duration in row.iteritems():
				if math.isnan(duration)==False:
					resource_info[activity] = duration
			resource=Resource(resource_name, resource_info)
			resource_set.append(resource)
		self.resource_set = resource_set

	def update_ongoing_instances(self,t):
		if t in self.release_schedule.keys():
			self.ongoing_instance += self.release_schedule[t]

	def update_completes(self):
		for i in self.ongoing_instance:
			if i.check_finished()==True:
				self.ongoing_instance.remove(i)
				self.completes.append(i)


	def update_w_resource(self, t):
		for j in self.resource_set:
			if j.get_next_ts() <= t:
				if j not in self.w_resource:
					self.w_resource.append(j)

	def update_w_instance(self, t):
		for i in self.ongoing_instance:
			if i.get_next_ts() <= t:
				if i not in self.w_instance:
					self.w_instance.append(i)

	def assign_res(self, t):
		for i in self.w_instance:
			next_act = i.get_next_act()
			for j in self.w_resource:
				#print('{} is assigned to {}'.format(i,j))
				if next_act in j.get_skills():
					next_ts = t+j.get_performance(next_act)
					event = (i.name, next_act, j.name, t, next_ts)
					self.eventlist.append(event)
					#self.p.hbar(y=[i.name], height=0.5,left=[t], right=[duration], line_color='red', line_width=2)
					i.set_next_ts(next_ts)
					i.set_next_act()
					j.set_next_ts(next_ts)
					"""
					if j.name=='Resource14':
						print("{} assigned to Resource14 at {} and finish at {}".format(i, t, duration))
					"""
					self.w_instance.remove(i)
					self.w_resource.remove(j)
					break
	def prod_hbar(self):
		TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"
		self.p = figure(y_range=self.total_caseid,sizing_mode='stretch_both', tools=TOOLS, toolbar_location='below')
	"""
	p = figure(y_range=CASE_ID,sizing_mode='stretch_both', tools=TOOLS, toolbar_location='below')
	colormap = color_list_generator(eventlog,'RESOURCE')
	p.hbar(y=[patient], height=0.5,left=[assigned_patient[patient]['start']], right=[available_times[patient]], color=colormap[assigned_resource])
	show(p)
	"""

	def simulate(self):
		t=0
		while self.total_ongoing_instance != len(self.completes):
		#for t in range(self.endpoint+6000):
			#if t%5 == 0:
			self.update_ongoing_instances(t)
			self.update_completes()
			self.update_w_instance(t)
			self.update_w_resource(t)
			self.assign_res(t)
			t+=1
			if t%100 == 0:
				print('num_ongoing: {}, num_w_instance: {}, num_w_resource: {}, num_completes: {}'.format(len(self.ongoing_instance), len(self.w_instance), len(self.w_resource), len(self.completes)))
		print("finish at {}".format(t-1))

		print(self.total_ongoing_instance)
		print("ongoing_instance: {}".format(self.ongoing_instance))
		print("completes: {}".format(self.completes))
		print(len(self.completes))
		print(len(self.w_instance))

	def prod_eventlog(self, start_point, columns=['CASE_ID', 'Activity', 'Resource', 'Start', 'Complete']):
		eventlog = pd.DataFrame.from_records(self.eventlist, columns=columns)
		eventlog['StartTimestamp']=start_point
		eventlog['CompleteTimestamp']=start_point
		eventlog['StartTimestamp'] = pd.to_datetime(eventlog['StartTimestamp'], format = '%Y-%m-%d %H:%M:%S', errors='ignore')
		eventlog['StartTimestamp'] += pd.to_timedelta(eventlog['Start'], unit='m')
		eventlog['CompleteTimestamp'] = pd.to_datetime(eventlog['CompleteTimestamp'], format = '%Y-%m-%d %H:%M:%S', errors='ignore')
		eventlog['CompleteTimestamp'] += pd.to_timedelta(eventlog['Complete'], unit='m')
		eventlog.sort_values(['CASE_ID', 'StartTimestamp'], inplace=True)
		eventlog = Eventlog(eventlog)
		return eventlog

	def generate_weight(self, eventlog):
		eventlog['weight'] = 0
		for case in eventlog.get_caseids():
			randint = np.random.randint(10)
			randint+=1
			eventlog.loc[eventlog['CASE_ID']==case, 'weight'] = randint
		eventlog['Duration'] = eventlog['Complete']-eventlog['Start']
		return eventlog






if __name__=='__main__':
	resource_info_name = '0806_1'
	count_list = [90]
	# count_list = [120]
	#testlog
	for count in count_list:
		Gen = LogGenerator(mode='test', endpoint=180, count=count, res_path="../sample_data/artificial/new_resource_{}.csv".format(resource_info_name))
		Gen.simulate()
		#show(Gen.p)
		eventlog = Gen.prod_eventlog(start_point='2018-12-01 00:00:00')
		eventlog = Gen.generate_weight(eventlog)

		i=0
		check=False
		removes = list()
		for row in eventlog.itertuples():
			if check == True:
				if row.Activity=="Discharge":
					removes.append(row.Index)
					check=False
				else:
					check=False
			else:
				if row.Activity == "Discharge":
					check=True
		eventlog = eventlog.loc[~eventlog.index.isin(removes)]
		print(eventlog)


		eventlog.to_csv('../sample_data/artificial/testlog_{}_{}.csv'.format(resource_info_name,count))


	"""
	#traininglog
	Gen = LogGenerator(7*24*60,res_path="../sample_data/new_resource_{}.csv".format(resource_info_name))
	Gen.simulate()
	eventlog = Gen.prod_eventlog(start_point='2018-12-01 00:00:00')
	eventlog = Gen.generate_weight(eventlog)
	eventlog.to_csv('../result/traininglog_{}.csv'.format(resource_info_name))
	"""