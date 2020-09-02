import config
from feature_generator import FeatureGenerator

import keras
import pickle
from datetime import datetime
from model import net

accuracy_values = list()
accuracy_sum = 0.0
accuracy_value = 0.0
precision_values = list()
precision_sum = 0.0
precision_value = 0.0
recall_values = list()
recall_sum = 0.0
recall_value = 0.0
f1_values = list()
f1_sum = 0.0
f1_value = 0.0
training_time_seconds = list()

args = ""

if __name__ == '__main__':
	args = config.load()

	level = args.inter_case_level
	#filename = req['name']

	filename = args.data_dir + args.data_set
	model_name = args.data_set + args.task

	contextual_info = args.contextual_info
	if args.task == 'next_activity':
		loss = 'categorical_crossentropy'
		regression = False
	elif args.task == 'next_timestamp':
		loss = 'mae'
		regression = True

	batch_size = args.batch_size_train
	num_folds = args.num_folds

    # load data
	FG = FeatureGenerator()
	df = FG.create_initial_log(filename)



	#split train and test
	#train_df, test_df = FG.train_test_split(df, 0.7, 0.3)
	train_df = df
	test_df = train_df
	#create train
	train_df = FG.order_csv_time(train_df)
	train_df = FG.queue_level(train_df)
	train_df.to_csv('./training_data.csv')
	state_list = FG.get_states(train_df)
	train_X, train_Y_Event, train_Y_Time = FG.one_hot_encode_history(train_df, args.checkpoint_dir+args.data_set)
	if contextual_info:
		train_context_X = FG.generate_context_feature(train_df,state_list)
		model = net()
		if regression:
			model.train(train_X, train_context_X, train_Y_Time, regression, loss, batch_size=batch_size, num_folds=num_folds, model_name=model_name, checkpoint_dir=args.checkpoint_dir)
		else:
			model.train(train_X, train_context_X, train_Y_Event, regression, loss, batch_size=batch_size, num_folds=num_folds, model_name=model_name, checkpoint_dir=args.checkpoint_dir)
	else:
		model_name += '_no_context_'
		train_context_X = None
		model = net()
		if regression:
			model.train(train_X, train_context_X, train_Y_Time, regression, loss, batch_size=batch_size, num_folds=num_folds, model_name=model_name, checkpoint_dir=args.checkpoint_dir, context=contextual_info)
		else:
			model.train(train_X, train_context_X, train_Y_Event, regression, loss, batch_size=batch_size, num_folds=num_folds, model_name=model_name, checkpoint_dir=args.checkpoint_dir, context=contextual_info)
	"""
	test_df = FG.order_csv_time(test_df)
	test_df = FG.queue_level(test_df)
	test_state_list = FG.get_states(test_df)

	test_X, test_Y_Event, test_Y_Time = FG.one_hot_encode_history(test_df)
	test_context_X = FG.generate_context_feature(test_df,test_state_list)
	"""
	test_X, test_Y_Event, test_Y_Time = train_X, train_Y_Event, train_Y_Time
	test_context_X = train_context_X
	test_X = test_X[500]
	test_context_X = test_context_X[500]
	test_Y_Event = test_Y_Event[500]
	#MC_pred, MC_uncertainty = model.predict(test_X, test_context_X, test_Y_Event)

	"""
	if args.dnn_architecture == 0:
		# train a 2-layer LSTM with one shared layer
		main_input = keras.layers.Input(shape=(sequence_max_length, num_features_all), name='main_input')
		# the shared layer
		l1 = keras.layers.recurrent.LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, dropout=0.2)(main_input)
		b1 = keras.layers.normalization.BatchNormalization()(l1)
		# the layer specialized in activity prediction
		l2_1 = keras.layers.recurrent.LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(b1)
		b2_1 = keras.layers.normalization.BatchNormalization()(l2_1)
		# the layer specialized in time prediction
		l2_2 = keras.layers.recurrent.LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(b1)
		b2_2 = keras.layers.normalization.BatchNormalization()(l2_2)

	context_shape = context_X.shape
	auxiliary_input = keras.layers.Input(shape=(context_shape[1],), name='aux_input')
	b2_1 = keras.layers.concatenate([b2_1, auxiliary_input])
	b2_2 = keras.layers.concatenate([b2_2, auxiliary_input])


	event_output = keras.layers.core.Dense(num_features_activities, activation='softmax', kernel_initializer='glorot_uniform', name='event_output')(b2_1)
	time_output = keras.layers.core.Dense(1, kernel_initializer='glorot_uniform', name='time_output')(b2_2)

	model_suffix_prediction = keras.models.Model(inputs=[main_input, auxiliary_input], outputs=[event_output, time_output])

	opt = keras.optimizers.Nadam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, schedule_decay=0.004, clipvalue=3)
	#opt = keras.optimizers.Adam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	#opt = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
	#opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0.004, clipvalue=3)
	#opt = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
	#opt = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)

	model_suffix_prediction.compile(loss={'event_output':'categorical_crossentropy', 'time_output':'mae'}, optimizer=opt)
	early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
	model_checkpoint = keras.callbacks.ModelCheckpoint('%smodel_suffix_prediction_.h5' % (args.checkpoint_dir), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
	lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
	model_suffix_prediction.summary()

	start_training_time = datetime.now()
	model_suffix_prediction.fit([X, context_X], {'event_output':Y_Event, 'time_output':Y_Time}, validation_split=1/args.num_folds, verbose=1, callbacks=[early_stopping, model_checkpoint, lr_reducer], batch_size=batch_size, epochs=args.dnn_num_epochs)
	training_time = datetime.now() - start_training_time
	"""

