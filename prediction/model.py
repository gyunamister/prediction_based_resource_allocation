# Copyright 2016, Yarin Gal, All rights reserved.
# This code is based on the code by Jose Miguel Hernandez-Lobato used for his
# paper "Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks".

import warnings
warnings.filterwarnings("ignore")

import math
from scipy.misc import logsumexp
import numpy as np

import keras
from keras.regularizers import l2
from keras import Input
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Dense
from keras import Model
from keras.models import load_model

import time



class net:
    def __init__(self):
        pass

    def train(self, X_train, X_train_ctx, y_train, regression, loss, n_epochs = 100,
        normalize = False, y_normalize=False, tau = 1.0, dropout = 0.05, batch_size= 128, context=True, num_folds=10, model_name='predictor', checkpoint_dir='./checkpoints/'):

        """
            Constructor for the class implementing a Bayesian neural network
            trained with the probabilistic back propagation method.
            @param X_train      Matrix with the features for the training data.
            @param y_train      Vector with the target variables for the
                                training data.
            @param n_epochs     Numer of epochs for which to train the
                                network. The recommended value 40 should be
                                enough.
            @param normalize    Whether to normalize the input features. This
                                is recommended unles the input vector is for
                                example formed by binary features (a
                                fingerprint). In that case we do not recommend
                                to normalize the features.
            @param tau          Tau value used for regularization
            @param dropout      Dropout rate for all the dropout layers in the
                                network.
        """

        # We normalize the training data to have zero mean and unit standard
        # deviation in the training set if necessary
        """
        if normalize:
            self.std_X_train_ctx = np.std(X_train_ctx, 0)
            self.std_X_train_ctx[ self.X_train_ctx == 0 ] = 1
            self.mean_X_train_ctx = np.mean(X_train_ctx, 0)
        else:
            self.std_X_train_ctx = np.ones(X_train_ctx.shape[ 1 ])
            self.mean_X_train_ctx = np.zeros(X_train_ctx.shape[ 1 ])

        X_train_ctx = (X_train_ctx - np.full(X_train_ctx.shape, self.mean_X_train_ctx)) / \
            np.full(X_train_ctx.shape, self.std_X_train_ctx)
        """
        if y_normalize:
            self.mean_y_train = np.mean(y_train)
            self.std_y_train = np.std(y_train)

            y_train_normalized = (y_train - self.mean_y_train) / self.std_y_train
            y_train_normalized = np.array(y_train_normalized, ndmin = 2).T
        else:
            if len(y_train.shape)==1:
                y_train_normalized = np.array(y_train, ndmin = 2).T
            else:
                y_train_normalized = y_train


        # We construct the network
        N = X_train.shape[0]
        batch_size = batch_size
        num_folds = num_folds

        inputs = Input(shape=(X_train.shape[1], X_train.shape[2]), name='main_input')
        inter = Dropout(dropout)(inputs, training=True)
        inter = LSTM(30, recurrent_dropout=dropout, return_sequences=True)(inputs, training=True)
        inter = Dropout(dropout)(inter, training=True)
        inter = LSTM(30)(inter, training=True)
        inter = Dropout(dropout)(inter, training=True)

        if context==True:
            context_shape = X_train_ctx.shape
            auxiliary_input = Input(shape=(context_shape[1],), name='aux_input')
            aux_inter = Dropout(dropout)(auxiliary_input, training=True)

            inter = keras.layers.concatenate([inter, aux_inter])
            inter = Dropout(dropout)(inter, training=True)

            if regression:
                outputs = Dense(y_train_normalized.shape[1], )(inter)
            else:
                outputs = Dense(y_train_normalized.shape[1], activation='softmax')(inter)
            model = Model(inputs=[inputs,auxiliary_input], outputs=outputs)

            model.compile(loss=loss, optimizer='adam')
            early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
            model_checkpoint = keras.callbacks.ModelCheckpoint('%smodel_%s_.h5' % (checkpoint_dir, model_name), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
            lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
            # We iterate the learning process
            start_time = time.time()
            model.fit([X_train,X_train_ctx], y_train_normalized, batch_size=batch_size, nb_epoch=n_epochs, verbose=2, validation_split=1/num_folds, callbacks=[early_stopping, model_checkpoint, lr_reducer])
        else:
            if regression:
                outputs = Dense(y_train_normalized.shape[1], )(inter)
            else:
                outputs = Dense(y_train_normalized.shape[1], activation='softmax')(inter)
            model = Model(inputs=inputs, outputs=outputs)

            model.compile(loss=loss, optimizer='adam')
            early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
            model_checkpoint = keras.callbacks.ModelCheckpoint('%smodel_%s_.h5' % (checkpoint_dir, model_name), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
            lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
            # We iterate the learning process
            start_time = time.time()
            model.fit(X_train, y_train_normalized, batch_size=batch_size, nb_epoch=n_epochs, verbose=2, validation_split=1/num_folds, callbacks=[early_stopping, model_checkpoint, lr_reducer])

        self.model = model
        self.tau = tau
        self.running_time = time.time() - start_time

        # We are done!

    def load(self, checkpoint_dir, model_name):
        model = load_model('%smodel_%s_.h5' % (checkpoint_dir, model_name))
        self.model = model

    def predict(self, X_test, X_test_ctx=None, context=True):

        """
            Function for making predictions with the Bayesian neural network.
            @param X_test   The matrix of features for the test data


            @return m       The predictive mean for the test target variables.
            @return v       The predictive variance for the test target
                            variables.
            @return v_noise The estimated variance for the additive noise.
        """

        X_test = np.array(X_test, ndmin = 3)


        # We normalize the test set
        #X_test_ctx = (X_test_ctx - np.full(X_test_ctx.shape, self.mean_X_train_ctx)) /    np.full(X_test_ctx.shape, self.std_X_train_ctx)

        # We compute the predictive mean and variance for the target variables
        # of the test data

        model = self.model
        """
        standard_pred = model.predict([X_test, X_test_ctx], batch_size=500, verbose=1)
        standard_pred = standard_pred * self.std_y_train + self.mean_y_train
        rmse_standard_pred = np.mean((y_test.squeeze() - standard_pred.squeeze())**2.)**0.5
        """
        T = 10
        if context==True:
            X_test_ctx = np.array(X_test_ctx, ndmin=2)
            Yt_hat = np.array([model.predict([X_test, X_test_ctx], batch_size=1, verbose=0) for _ in range(T)])
        else:
            Yt_hat = np.array([model.predict(X_test, batch_size=1, verbose=0) for _ in range(T)])
        #Yt_hat = Yt_hat * self.std_y_train + self.mean_y_train
        regression=False
        if regression:
            MC_pred = np.mean(Yt_hat, 0)
            MC_uncertainty = np.std(Yt_hat, 0)
        else:
            MC_pred = np.mean(Yt_hat, 0)
            MC_uncertainty = list()
            for i in range(Yt_hat.shape[2]):
                MC_uncertainty.append(np.std(Yt_hat[:,:,i].squeeze(),0))
        #rmse = np.mean((y_test.squeeze() - MC_pred.squeeze())**2.)**0.5

        # We compute the test log-likelihood
        """
        ll = (logsumexp(-0.5 * self.tau * (y_test[None] - Yt_hat)**2., 0) - np.log(T)
            - 0.5*np.log(2*np.pi) + 0.5*np.log(self.tau))
        test_ll = np.mean(ll)
        """
        # We are done!
        return MC_pred, MC_uncertainty