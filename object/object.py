import sys
import os
from pathlib import Path
import time

import numpy as np
import math

p = Path(__file__).resolve().parents[2]
sys.path.append(os.path.abspath(str(p)))


class Instance(object):
	#prediction 수행 -> predicted_next_act, time
	#prediction을 위한 input vector
	def __init__(self, name, weight, *args, **kwargs):
		super(Instance, self).__init__()
		if 'act_sequence' in kwargs:
			act_sequence = kwargs['act_sequence']
			self.set_act_sequence(act_sequence)
			#self.next_act = self.seq[0]
		elif 'initial_activity' in kwargs:
			self.next_act = kwargs['initial_activity']
		else:
			raise AttributeError('Either sequence or initial_activity should be given.')

		if 'res_sequence' in kwargs:
			res_sequence = kwargs['res_sequence']
			self.set_res_sequence(res_sequence)

		if 'dur_sequence' in kwargs:
			dur_sequence = kwargs['dur_sequence']
			self.set_dur_sequence(dur_sequence)

		if 'release_time' in kwargs:
			release_time = kwargs['release_time']
			self.set_release_time(release_time)
		else:
			release_time = False

		if 'initial_index' in kwargs:
			self.cur_index = kwargs['initial_index']
			self.first = True
		else:
			self.cur_index = -1

		self.name = name
		self.next_actual_act = self.act_sequence[self.cur_index+1]
		self.next_pred_act = self.act_sequence[self.cur_index+1]
		self.next_actual_ts = release_time
		self.next_pred_ts = release_time
		self.weight=weight
		self.initial_weight = weight
		self.status = True
		self.pred_act_dur_dict = dict()
		self.pred_time_list = list()
		self.cur_act_trace = list()
		self.cur_res_trace = list()
		self.first = True
		print("ID: {}-{}, Seq: {}, Cur: {}, Released: {}".format(self.name, self.cur_index, self.act_sequence, self.next_actual_act, release_time))

	def __str__(self):
		return self.name

	def __repr__(self):
		return self.name

	@classmethod
	def set_model_next_act(cls, pred_model):
		cls.model_next_act = pred_model

	@classmethod
	def set_model_next_time(cls, pred_model):
		cls.model_next_time = pred_model

	@classmethod
	def set_est_next_time(cls, pred_model):
		cls.est_next_time = pred_model

	@classmethod
	def set_activity_list(cls, activity_list):
		cls.activity_list = activity_list

	@classmethod
	def set_resource_list(cls, resource_list):
		cls.resource_list = resource_list

	@classmethod
	def set_act_char_to_int(cls, act_char_to_int):
		#print("act_char_to_int: {}".format(act_char_to_int))
		cls.act_char_to_int = act_char_to_int

	@classmethod
	def set_act_int_to_char(cls, act_int_to_char):
		#print("act_int_to_char: {}".format(act_int_to_char))
		cls.act_int_to_char = act_int_to_char

	@classmethod
	def set_res_char_to_int(cls, res_char_to_int):
		#print("res_char_to_int: {}".format(res_char_to_int))
		cls.res_char_to_int = res_char_to_int

	@classmethod
	def set_res_int_to_char(cls, res_int_to_char):
		#print("res_int_to_char: {}".format(res_int_to_char))
		cls.res_int_to_char = res_int_to_char

	@classmethod
	def set_maxlen(cls, maxlen):
		cls.maxlen = maxlen

	def old_update_x(self, act_trace, res_trace):
		for act, res in zip(act_trace, res_trace):
			activity_trace_i = act_trace[:i+1]
			resource_trace_i = res_trace[:i+1]
			act_int_encoded_X = [self.act_char_to_int[activity] for activity in activity_trace_i]
			res_int_encoded_X = [self.res_char_to_int[resource] for resource in resource_trace_i]

			# one hot encode X
			onehot_encoded_X = list()
			for act_value, res_value in zip(act_int_encoded_X, res_int_encoded_X):
				act_letter = [0 for _ in range(len(self.activity_list))]
				res_letter = [0 for _ in range(len(self.resource_list))]
				act_letter[act_value] = 1
				res_letter[res_value] = 1
				letter = act_letter + res_letter
				onehot_encoded_X.append(letter)
			num_act_res = len(self.activity_list)+len(self.resource_list)
			#zero-pad
			while len(onehot_encoded_X) != self.maxlen:
				onehot_encoded_X.insert(0, [0]*num_act_res)
		onehot_encoded_X = [onehot_encoded_X]
		onehot_encoded_X = np.asarray(onehot_encoded_X)
		return onehot_encoded_X

	def update_x(self, act_trace, res_trace):
		int_encoded_act = [self.act_char_to_int[act] for act in act_trace]
		int_encoded_res = [self.res_char_to_int[res] for res in res_trace]
		num_act, num_res = len(self.activity_list), len(self.resource_list)
		# one hot encode X
		onehot_encoded_X = list()
		for act_int, res_int in zip(int_encoded_act, int_encoded_res):
		    onehot_encoded_act = [0 for _ in range(num_act)]
		    onehot_encoded_act[act_int] = 1
		    onehot_encoded_res = [0 for _ in range(num_res)]
		    onehot_encoded_res[res_int] = 1
		    onehot_encoded = onehot_encoded_act + onehot_encoded_res
		    onehot_encoded_X.append(onehot_encoded)

		#zero-pad
		while len(onehot_encoded_X) != self.maxlen:
		    onehot_encoded_X.insert(0, [0]*(num_act+num_res))
		onehot_encoded_X = np.asarray(onehot_encoded_X)
		return onehot_encoded_X

	def predict_next_act(self, queue, context= True, ):
		time1 = time.time()
		"""
		if self.cur_index > 0:
			act_trace = self.act_sequence[:self.cur_index-1] + [cur_act]
			res_trace = self.res_sequence[:self.cur_index-1] + [resource]
		else:
			act_trace = [cur_act]
			res_trace = [resource]
		"""
		X = self.update_x(self.cur_act_trace, self.cur_res_trace)
		if context==True:
			context_X = np.array(list(queue.values()))
			pred_vector, conf_vector = self.model_next_act.predict(X, context_X)
		else:
			pred_vector, conf_vector = self.model_next_act.predict(X, context=False)
		#pred_vector, conf_vector = self.model_next_act.predict(X, context=False)
		pred_index = np.argmax(pred_vector,axis=1)[0]
		pred_next_act = self.act_int_to_char[pred_index]
		conf = conf_vector[pred_index]

		time2 = time.time()
		self.pred_time_list.append(time2-time1)
		return pred_next_act, conf

	def estimate_next_time(self, queue, context= True, pred_act=False, resource=False):
		"""
		if self.cur_index > 0:
			act_trace = self.act_sequence[:self.cur_index-1] + [cur_act]
			res_trace = self.res_sequence[:self.cur_index-1] + [resource]
		else:
			act_trace = [cur_act]
			res_trace = [resource]
		"""
		if pred_act!= False and resource!=False:
			X = self.update_x(self.cur_act_trace + [pred_act], self.cur_res_trace+ [resource])
		else:
			X = self.update_x(self.cur_act_trace, self.cur_res_trace)
		if context==True:
			context_X = np.array(list(queue.values()))
			dur_pred, conf = self.model_next_time.predict(X, context_X)

		else:
			dur_pred, conf = self.model_next_time.predict(X, context=False)

		pred_dur = math.floor(dur_pred[0][0])
		if pred_dur <= 0:
			pred_dur = 1
		return pred_dur, conf[0]

	def predict_next_time(self, queue, context= True, pred_act=False, resource=False):
		time1 = time.time()
		"""
		if self.cur_index > 0:
			act_trace = self.act_sequence[:self.cur_index-1] + [cur_act]
			res_trace = self.res_sequence[:self.cur_index-1] + [resource]
		else:
			act_trace = [cur_act]
			res_trace = [resource]
		"""
		if pred_act!= False and resource!=False:
			X = self.update_x(self.cur_act_trace + [pred_act], self.cur_res_trace+ [resource])
		else:
			X = self.update_x(self.cur_act_trace, self.cur_res_trace)
		if context==True:
			context_X = np.array(list(queue.values()))
			dur_pred, conf = self.model_next_time.predict(X, context_X)

		else:
			dur_pred, conf = self.model_next_time.predict(X, context=False)

		pred_dur = math.floor(dur_pred[0][0])
		if pred_dur <= 0:
			pred_dur = 1
		time2 = time.time()
		self.pred_time_list.append(time2-time1)
		return pred_dur, conf[0]

	def update_res_history(self, resource):
		self.res_sequence.append(resource)

	def get_name(self):
		return self.name

	def get_cur_actual_act(self):
		if self.first == True:
			return False
		else:
			return self.cur_actual_act

	def get_next_pred_act(self):
		return self.next_pred_act

	def get_next_act_uncertainty(self):
		return self.next_act_uncertainty

	def get_next_pred_ts(self):
		return self.next_pred_ts

	def get_next_ts_uncertainty(self, res):
		return self.pred_act_dur_dict[res][1]

	def get_next_actual_ts(self):
		return self.next_actual_ts

	def get_next_actual_act(self):
		return self.next_actual_act

	def get_next_actual_act_dur(self):
		return self.next_actual_act_dur

	def get_pred_act_dur_dict(self):
		return self.pred_act_dur_dict

	def get_pred_act_dur(self, res):
		try:
			return self.pred_act_dur_dict[res][0]
		except KeyError:
			print("ERROR: {} is not in the dict".format(res.get_name()))
			return 5

	def get_release_time(self):
		return self.release_time

	def set_act_sequence(self, act_sequence):
		self.act_sequence = act_sequence

	def set_res_sequence(self, res_sequence):
		self.res_sequence = res_sequence

	def set_dur_sequence(self, dur_sequence):
		self.dur_sequence = dur_sequence

	def set_release_time(self, release_time):
		self.release_time = release_time

	def set_next_actual_act(self):
		if self.cur_index < len(self.act_sequence)-1:
			self.next_actual_act = self.act_sequence[self.cur_index+1]

	def set_next_actual_ts(self, next_actual_ts):
		self.next_actual_ts = next_actual_ts

	def set_pred_act_dur(self, res, pred_act_dur, conf):
		self.pred_act_dur_dict[res] = [pred_act_dur, conf]



	def clear_pred_act_dur(self):
		self.pred_act_dur_dict = dict()

	def set_next_pred_act(self, next_pred_act):
		self.next_pred_act = next_pred_act

	def set_next_act_uncertainty(self, next_act_uncertainty):
		self.next_act_uncertainty = next_act_uncertainty

	def set_next_ts_uncertainty(self, res, next_ts_uncertainty):
		self.pred_act_dur_dict[res][1] = next_ts_uncertainty

	def set_next_pred_ts(self, next_pred_ts):
		self.next_pred_ts = next_pred_ts

	def update_actuals(self, t, res, mode, act_res_mat, queue):
		# set next ts
		self.cur_index+=1
		self.first =False
		if self.cur_index > len(self.act_sequence)-1:
			print("{} exceed the limit".format(self.get_name()))
			return
		# set current act
		self.cur_actual_act = self.act_sequence[self.cur_index]

		# set next timestamp
		self.set_next_pred_ts(t+self.get_pred_act_dur(res))
		if mode == 'test':
			self.set_next_actual_ts(t+int(act_res_mat[self.cur_actual_act][res.get_name()]))
		else:
			# (CHANGED)
			self.set_next_actual_ts(self.get_next_pred_ts())
			# next_est_dur, next_time_uncertainty = self.estimate_next_time(queue, context=True, pred_act=self.cur_actual_act, resource=res.get_name())
			# self.set_next_actual_ts(t+next_est_dur)

		# set current seq
		if self.cur_index < len(self.act_sequence)-1:
			self.cur_act_trace = self.act_sequence[:self.cur_index+1]
			self.cur_res_trace = self.res_sequence[:self.cur_index+1]

		# set next act
		if self.cur_index < len(self.act_sequence)-1:
			self.next_actual_act = self.act_sequence[self.cur_index+1]

		self.set_status(False)
		self.update_res_history(res.get_name())
		print("{}'s {}:Process {} pred_till {} actual_till {}, resource {}, next_act: {},activity_trace: {}, resource_trace: {}".format(self.get_name(), self.cur_index, self.get_cur_actual_act(), self.get_next_pred_ts(),self.get_next_actual_ts(), res.get_name(), self.next_actual_act, self.cur_act_trace, self.cur_res_trace))

	def get_weight(self):
		return self.weight

	def update_weight(self):
		self.weight += 10000
		return self.weight

	def reset_weight(self):
		self.weight = self.initial_weight

	def set_status(self, status):
		self.status = status

	def get_status(self):
		return self.status

	def check_finished(self, t):
		# index가 sequence length와 같아지면 종
		if self.cur_index >= len(self.act_sequence)-1:
			return True
		else:
			return False

	def set_weighted_comp(self):
		self.weighted_comp = (self.get_next_actual_ts()-self.release_time)*self.initial_weight

	def get_weighted_comp(self):
		return self.weighted_comp

	def get_pred_time_list(self):
		return self.pred_time_list



class Resource(object):
	def __init__(self, name, skills, *args, **kwargs):
		super(Resource, self).__init__()
		self.name = name
		self.skills = skills
		self.next_pred_ts=0
		self.next_actual_ts = 0
		self.next_ts_uncertainty = 1
		self.status=True
		self.dur_dict = dict()

	def __str__(self):
		return self.name

	def __repr__(self):
		return self.name

	def get_name(self):
		return self.name

	def get_skills(self):
		return self.skills

	def get_next_pred_ts(self):
		return self.next_pred_ts

	def get_next_ts_uncertainty(self):
		return self.next_ts_uncertainty

	def get_next_actual_ts(self):
		return self.next_actual_ts

	def set_next_actual_ts(self, next_actual_ts):
		self.next_actual_ts = next_actual_ts

	def set_next_pred_ts(self, next_pred_ts):
		self.next_pred_ts = next_pred_ts

	def set_next_ts_uncertainty(self, conf):
		self.next_ts_uncertainty = conf

	def set_status(self, status):
		self.status = status

	def get_status(self):
		return self.status



