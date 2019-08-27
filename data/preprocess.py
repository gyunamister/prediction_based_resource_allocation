import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import math

p = Path(__file__).resolve().parents[2]
sys.path.append(os.path.abspath(str(p)))
from PyProM.src.data.Eventlog import Eventlog

def remove_micro_seconds(x):
	if len(x) > 19:
		x = x[:19]
	return x

def to_minute(x):
	if np.isnan(x.seconds):
		return x
	#return int(x.seconds/60)
	return math.ceil(x.seconds/60)

if __name__=='__main__':
	path = './sample_data/BPI_Challenge_2012.csv'
	eventlog = Eventlog.from_txt(path, sep=',')
	eventlog = eventlog.assign_caseid('Case ID')
	eventlog = eventlog.assign_activity('Activity')
	#eventlog = eventlog.assign_resource('Resource')
	eventlog['transition'] = eventlog['lifecycle:transition']
	eventlog['CompleteTimestamp'] = eventlog['Complete Timestamp']
	eventlog['CompleteTimestamp'] = eventlog['CompleteTimestamp'].apply(remove_micro_seconds)
	eventlog['CompleteTimestamp'] = eventlog['CompleteTimestamp'].str.replace('.', '/', regex=False)
	eventlog['Amount'] = eventlog['(case) AMOUNT_REQ']
	#eventlog = eventlog.assign_timestamp('Start Timestamp', name='Timestamp', format = '%Y/%m/%d %H:%M:%S')
	eventlog = eventlog.loc[(eventlog['Activity'].str.contains('W_', regex=False)) & ~(eventlog['Activity'].str.contains('SCHEDULE'))]
	caseid = ''
	temp_caseid=0
	start = False
	table = list()
	for row in eventlog.itertuples():
		if caseid != row.CASE_ID:
			caseid = row.CASE_ID
			temp_caseid += 1
			start = False
		if row.transition == 'START':
			if start == True:
				table.append(data)
			start = True
			data = list()
			data += [temp_caseid, row.Activity, row.Resource, row.CompleteTimestamp, row.Amount, '']
		if row.transition == 'COMPLETE':
			if start==True:
				data[-1] = row.CompleteTimestamp
			else:
				data = list()
				data += [temp_caseid, row.Activity, row.Resource, '', row.Amount, row.CompleteTimestamp]
			start = False
			table.append(data)
	headers = ['CASE_ID', 'Activity', 'Resource', 'StartTimestamp', 'Amount','CompleteTimestamp']
	df = pd.DataFrame(table, columns=headers)
	eventlog = Eventlog(df)
	eventlog = eventlog.assign_timestamp(name='StartTimestamp', new_name='StartTimestamp', _format = '%Y.%m.%d %H:%M:%S', errors='raise')
	eventlog = eventlog.assign_timestamp(name='CompleteTimestamp', new_name='CompleteTimestamp', _format = '%Y/%m/%d %H:%M:%S', errors='raise')
	eventlog['Duration'] = (eventlog['CompleteTimestamp'] - eventlog['StartTimestamp']).apply(to_minute)

	eventlog.dropna(subset=['Resource', 'StartTimestamp', 'CompleteTimestamp'],inplace=True)


	#100회 미만 제외
	resource_count = eventlog.groupby('Resource').CASE_ID.count()
	valid_resource = resource_count[resource_count >= 100].astype(int)
	valid_resource_list = list(valid_resource.index)
	eventlog = eventlog.loc[eventlog['Resource'].isin(valid_resource_list)]

	#20회 초과 이벤트 제외
	event_count = eventlog.groupby('CASE_ID').Activity.count()
	invalid_case = event_count[event_count > 20].astype(int)
	invalid_case_list = list(invalid_case.index)
	eventlog = eventlog.loc[~eventlog['CASE_ID'].isin(invalid_case_list)]

	#특정 날짜(scheduling 대상)에 event가 있는 모든 instance 추출
	eventlog['StartDate'] = eventlog['StartTimestamp'].dt.date
	eventlog['CompleteDate'] = eventlog['CompleteTimestamp'].dt.date
	#d='2012.3-10'
	#d_list = ['2012.3.1', '2012.3.2', '2012.3.3','2012.3.4','2012.3.5','2012.3.6','2012.3.7','2012.3.8','2012.3.9','2012.3.10', '2012.3.11','2012.3.12','2012.3.13', '2012.3.14', '2012.3.15']
	d_list = ['2012-03-01', '2012-03-02', '2012-03-03','2012-03-04','2012-03-05','2012-03-06','2012-03-07','2012-03-08','2012-03-09','2012-03-10', '2012-03-11','2012-03-12','2012-03-13', '2012-03-14', '2012-03-15']
	def correct_activity_name(x):
		return x.replace("_", "-")
	for d in d_list:
		target_date=pd.to_datetime(d).date()
		valid_case = eventlog.loc[eventlog['StartDate']==target_date,'CASE_ID'].unique()
		sublog = eventlog.loc[eventlog['CASE_ID'].isin(valid_case)]

		#target date 이후 이벤트 삭제 (하루치만)
		sublog = sublog.loc[sublog['StartDate']<=target_date]

		#weight 배정
		max_amount = sublog['Amount'].max()
		print(max_amount)
		#custom_bucket_array = np.linspace(0, max_amount, 10)
		labels = [x+1 for x in range(10)]
		sublog['weight'] = pd.qcut(df['Amount'], 10, labels=labels)
		sublog['Activity'] = sublog['Activity'].apply(correct_activity_name)

		sublog.to_csv('../result/modi_BPI_2012_{}.csv'.format(d))
		print(sublog)
		print(len(sublog['CASE_ID'].unique()))
