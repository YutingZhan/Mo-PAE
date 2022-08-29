#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

# 2022.8.9
# Yuting Zhan

import numpy as np
from numpy import zeros
from numpy import loadtxt
import pandas as pd

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

LAMBDA_1 = -0.1
LAMBDA_2 = 0.8
LAMBDA_3 = -0.1
LEARNING_RATE = 0.0003

CONTEXT = 10
EPOCHS = 1000
BATCH_SIZE = 128
HISTORY = str(LAMBDA_1)+'_'+str(LAMBDA_2)+'_'+str(LAMBDA_3)+'_r'+str(EPOCHS)+'_his'+str(CONTEXT)+'_history.npy'
SAVED_MODEL = str(LAMBDA_1)+'_'+str(LAMBDA_2)+'_'+str(LAMBDA_3)+'_r'+str(EPOCHS)+'_his'+str(CONTEXT)+'_model.h5'
PRE_PNG = str(LAMBDA_1)+'_'+str(LAMBDA_2)+'_'+str(LAMBDA_3)+'_r'+str(EPOCHS)+'_his'+str(CONTEXT)+'_pre.png'
ID_PNG = str(LAMBDA_1)+'_'+str(LAMBDA_2)+'_'+str(LAMBDA_3)+'_r'+str(EPOCHS)+'_his'+str(CONTEXT)+'_id.png'
REC_PNG = str(LAMBDA_1)+'_'+str(LAMBDA_2)+'_'+str(LAMBDA_3)+'_r'+str(EPOCHS)+'_his'+str(CONTEXT)+'_rec.png'
PRE_TOPN_RESULT = str(LAMBDA_1)+'_'+str(LAMBDA_2)+'_'+str(LAMBDA_3)+'_r'+str(EPOCHS)+'_his'+str(CONTEXT)+'_accuracy.txt'
CSV_FILE = str(LAMBDA_1)+'_'+str(LAMBDA_2)+'_'+str(LAMBDA_3)+'_r'+str(EPOCHS)+'_his'+str(CONTEXT)+'_history.csv'
BEST_MODEL = str(LAMBDA_1)+'_'+str(LAMBDA_2)+'_'+str(LAMBDA_3)+'_r'+str(EPOCHS)+'_his'+str(CONTEXT)+'_model.h5'

# Base Director

DIR = './'
DATASET = DIR + 'GeoLifeBeijing_grid_train.csv' 
DATASET_TEST = DIR + 'GeoLifeBeijing_grid_test.csv'

BEST_MODEL = DIR +str(LAMBDA_1)+'_'+str(LAMBDA_2)+'_'+str(LAMBDA_3)+'_r'+str(EPOCHS)+'_his'+str(CONTEXT)+'_model.h5'

# Load dataset
df_train = pd.read_csv(DATASET)
df_test = pd.read_csv(DATASET_TEST)
df = [df_train, df_test]
df = pd.concat(df)
ids = df['label'].values
locations = df['loc'].values
unique_locations = pd.unique(locations)
total_locations = len(unique_locations)
unique_ids = pd.unique(ids)
total_ids = len(unique_ids)
id_index = pd.unique(ids)
id_index = pd.DataFrame(id_index).reset_index()
id_index.columns = ['newid', 'oldid']
id_index['newid'] = id_index['newid'] + 1
id_dict = dict(zip(id_index['oldid'], id_index['newid']))
df['label'] = df['label'].map(id_dict)
loc_index = pd.unique(locations)
loc_index = pd.DataFrame(loc_index).reset_index()
loc_index.columns = ['newloc', 'oldloc']
loc_index['newloc'] = loc_index['newloc'] + 1
loc_dict = dict(zip(loc_index['oldloc'], loc_index['newloc']))
df['loc'] = df['loc'].map(loc_dict)
df_dataset = df
print('Real Data Shape:', df_dataset.shape)

#Create the trainning sets of sequences with a lenght of INPUT_LOACTIONS
def get_data(X, Z, INPUT_POINTS):
  last_location = len(X)
  X_locations = []
  target_locations = [] # next points - ground truth
  target_ids = [] # ids - ground truth
  y = [] 
  z = []
  
  for i in range(last_location-INPUT_POINTS):
      X_locations.append(X[i:i+INPUT_POINTS])
      #represent the target location as a onehot for the softmax
      target_location = X[i+INPUT_POINTS]
      target_locations.append(target_location)
      target_location_onehot = np.zeros(total_locations)
      target_location_onehot[int(target_location)-1] = 1.0
      y.append(target_location_onehot)

      #represent the target id as onehot for the softmax
      target_id = Z[i]
      target_ids.append(target_id)
      target_id_onehot = np.zeros(total_ids)
      target_id_onehot[int(target_id)-1] = 1.0
      z.append(target_id_onehot)
  return X_locations, y, z

ids = df_dataset['label'].values
locations = df_dataset['loc'].values
timesteps = CONTEXT

X, y, z = get_data(X = locations, Z = ids, INPUT_POINTS = timesteps)
X = np.array(X)
y = np.array(y)
z = np.array(z)
n_features = 1
#X
def non_shuffling_train_test_split(X, y, z, test_size):
    i = int((1 - test_size) * X.shape[0]) + 1
    X_train, X_test = np.split(X, [i])
    y_train, y_test = np.split(y, [i])
    z_train, z_test = np.split(z, [i])
    return X_train, X_test, y_train, y_test, z_train, z_test

X_train, X_test, y_train, y_test, z_train, z_test = non_shuffling_train_test_split(X, y, z, 0.2)

print ('*'*80)
print ('Build Model......')
# define encoder
def encoder_model():
  model = tf.keras.Sequential()
  #model.add(Input(shape=(timesteps,n_features)))
  model.add(LSTM(512, activation='relu', return_sequences=True, input_shape=(n_features, timesteps))) #
  model.add(LSTM(256, activation='relu', return_sequences=False))
  return model
encoder = encoder_model()

def decoder_model():
  model = tf.keras.Sequential()
  model.add(RepeatVector(timesteps))
  model.add(LSTM(256, activation='relu', return_sequences=True))
  model.add(LSTM(512, activation='relu', return_sequences=True))
  model.add(TimeDistributed(Dense(n_features)))
  return model
decoder = decoder_model()

def id_recognizer_model():
  model = tf.keras.Sequential()
  model.add(Dense(256, activation='tanh'))
  model.add(Dropout(0.3))
  model.add(Dense(512, activation='tanh'))
  model.add(Dropout(0.3))
  model.add(Dense(total_ids, activation='softmax'))
  return model
id_recognizer = id_recognizer_model()

def next_prediction_model():
  model = tf.keras.Sequential()
  model.add(Dense(256, activation='tanh'))
  model.add(Dropout(0.3))
  model.add(Dense(512, activation='tanh'))
  model.add(Dropout(0.3))
  model.add(Dense(total_locations, activation='softmax'))
  return model
next_predictor = next_prediction_model()

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

loss_tracker = keras.metrics.Mean(name='loss')
loss1_tracker = keras.metrics.Mean(name='loss1')
loss1_1_tracker = keras.metrics.Mean(name='loss1_1')
loss2_tracker = keras.metrics.Mean(name='loss2')
loss3_tracker = keras.metrics.Mean(name='loss3')
mae1_metric = keras.metrics.MeanAbsoluteError(name='mae1')
mae2_metric = keras.metrics.MeanAbsoluteError(name='mae2')
mae3_metric = keras.metrics.MeanAbsoluteError(name='mae3')
acc1_metric = keras.metrics.CategoricalAccuracy(name='acc1')
acc2_metric = keras.metrics.CategoricalAccuracy(name='acc2')
acc3_metric = keras.metrics.CategoricalAccuracy(name='acc3')

@tf.function
def get_loss1(loss1, loss1_1):
  if (loss1 < 1):
     return loss1_1
  return loss1

class CustomModel(keras.Model):

  def __init__(self, encoder, decoder, id_recognizer, next_predictor):
    super(CustomModel, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.id_recognizer = id_recognizer 
    self.next_predictor = next_predictor
    self._set_inputs(tf.TensorSpec([None, 1, timesteps], tf.int64))

  def call(self, inputs):
    x1 = self.encoder(inputs)
    x2 = self.decoder(x1)
    x3 = self.next_predictor(x1)
    x4 = self.id_recognizer(x1)
    return x2, x3, x4

  def get_config(self):
    config = super(CustomModel, self).get_config()
    config.update({'encoder': self.encoder, 'decoder': self.decoder, 'id_recognizer': self.id_recognizer, 'next_predictor': self.next_predictor})
    return config

  def compile(self, model_optimizer):
    super(CustomModel, self).compile()
    self.model_optimizer = model_optimizer

  def train_step(self, data):
    # unpack the data. It structure depends on your model and on what to pass to fit()
    x, [y1, y2, y3] = data
    print ('here1:', x, y1, y2, y3)

    #Train the AE
    with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape, tf.GradientTape() as id_tape, tf.GradientTape() as next_tape:

      x = K.cast(x, dtype='float32')
      encoder_output = self.encoder(x, training =True)

      y1_pred = self.decoder(encoder_output, training = True)
      y1 = K.cast(y1, dtype='float32')
      print(y1)
      y1 = tf.expand_dims(y1, -1)
      print(y1)
      y1_pred = K.cast(y1_pred, dtype='float32')
      print(y1_pred)
      y1_pred = tf.expand_dims(y1_pred, 1)
      print(y1_pred)
      loss1 = tf.math.log(tf.math.reduce_mean(K.square(y1_pred - y1)))
      loss1_1 = tf.math.reduce_mean(K.square(y1_pred - y1))

      y2_pred = self.next_predictor(encoder_output, training = True)
      y2 = K.cast(y2, dtype='float32')
      print (y2)
      y2_pred = K.cast(y2_pred, dtype='float32')
      print (y2_pred)
      loss2 = tf.math.reduce_mean(keras.losses.categorical_crossentropy(y2, y2_pred, from_logits=True))
      loss2 = tf.math.log(loss2)

      y3_pred = self.id_recognizer(encoder_output, training = True)
      y3 = K.cast(y3, dtype='float32')
      print (y3)
      y3_pred = K.cast(y3_pred, dtype='float32')
      print (y3_pred)
      loss3 = tf.math.reduce_mean(keras.losses.categorical_crossentropy(y3, y3_pred, from_logits=True))

      loss1 = get_loss1(loss1, loss1_1)
      loss =  LAMBDA_1 * loss1 +  LAMBDA_2 * loss2 + LAMBDA_3 *loss3

    dec_gradients = dec_tape.gradient(loss1, self.decoder.trainable_variables)
    self.model_optimizer.apply_gradients(zip(dec_gradients, self.decoder.trainable_variables))

    next_gradients = next_tape.gradient(loss2, self.next_predictor.trainable_variables)
    self.model_optimizer.apply_gradients(zip(next_gradients, self.next_predictor.trainable_variables))

    id_gradients = id_tape.gradient(loss3, self.id_recognizer.trainable_variables)
    self.model_optimizer.apply_gradients(zip(id_gradients, self.id_recognizer.trainable_variables))

    enc_gradients = enc_tape.gradient(loss, self.encoder.trainable_variables)
    self.model_optimizer.apply_gradients(zip(enc_gradients, self.encoder.trainable_variables))
    
    loss_tracker.update_state(loss)
    loss1_tracker.update_state(loss1)
    loss2_tracker.update_state(loss2)
    loss3_tracker.update_state(loss3)
    mae1_metric.update_state(y1, y1_pred)
    acc2_metric.update_state(y2, y2_pred)
    acc3_metric.update_state(y3, y3_pred)

    return {'loss':loss_tracker.result(), 'mae1':mae1_metric.result(), 'loss1': loss1_tracker.result(), 'loss2':loss2_tracker.result(), 'loss3':loss3_tracker.result(), 'acc2':acc2_metric.result(), 'acc3':acc3_metric.result()}
  
  @property
  def metrics(self):
    # we list our 'Metrics' objects here so that reset_states() can be called automatically at the start of each epoch or at the start of evaluate()
    # if you don't implement this property, you have to call reset_states() yourself at the time of your choosing
    #return [loss_tracker, mae1_metric, mae2_metric, mae3_metric, loss1_tracker, loss2_tracker, loss3_tracker, acc1_metric, acc2_metric, acc3_metric]
    return [loss_tracker, mae1_metric, loss1_tracker, loss2_tracker, loss3_tracker, acc2_metric, acc3_metric]

new = CustomModel(encoder=encoder, decoder=decoder, id_recognizer=id_recognizer, next_predictor=next_predictor)
new.compile(model_optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))

print ('*'*80)
print ('Model Training......')

x_train = tf.expand_dims(X_train, 1)
x_test = tf.expand_dims(X_test, 1)
print (x_train.shape, x_test.shape)
checkpoint = ModelCheckpoint(BEST_MODEL, monitor='loss', verbose=0, save_best_only=True, save_weights_only=True, mode='min')
history = new.fit(x_train, [x_train, y_train, z_train], epochs=EPOCHS, batch_size=128, verbose=2, callbacks=[checkpoint])

CAE_model = CustomModel(encoder=encoder, decoder=decoder, id_recognizer=id_recognizer, next_predictor=next_predictor)
CAE_model.built = True
CAE_model.load_weights(BEST_MODEL)

print ('Training Finish......')
print ('*'*80)

print('Visualize the loss......')
history = history.history
np.save(HISTORY, history)

df_his = pd.DataFrame(history)
df_his.to_csv(CSV_FILE, header=True, index=False)

X_pre, y_pre, z_pre = CAE_model.predict(x_train, verbose = 0)
X_pre_test, y_pre_test, z_pre_test = CAE_model.predict(x_test, verbose = 0)

# Reconstruction Part
print ('*'*80)
print ('Reconstruction Part')

def get_predictions(ypre):
  predictions = []
  for i in range(len(ypre)):
    predictions.append(ypre[i][0][0])
  return predictions

def re_temporalize(yhat):
    loc_id = []
    for i in range(len(yhat)):
      loc_id.append(yhat[i][0][0])
    return loc_id

predictions = np.round(get_predictions(X_pre), decimals=0)
predictions_test = np.round(get_predictions(X_pre_test), decimals=0)

locid_pre = np.round(re_temporalize(X_pre))
print (locid_pre, len(locid_pre))

locid_ori = locations[:len(x_train)]
print(locid_ori, len(locid_ori))

locid_pre_test = np.round(re_temporalize(X_pre_test))
print (locid_pre_test, len(locid_pre_test))

locid_ori_test = locations[len(x_train):len(locations)-CONTEXT]
print(locid_ori_test, len(locid_ori_test))

# calculate minkowski distance
def minkowski_distance(a, b, p):
	return sum(abs(e1-e2)**p for e1, e2 in zip(a,b))**(1/p)

# calculate distance (p=1)
man_dist = minkowski_distance(locid_pre, locid_ori, 1)
man_dist_test = minkowski_distance(locid_pre_test, locid_ori_test, 1)

# calculate distance (p=2)
euc_dist = minkowski_distance(locid_pre, locid_ori, 2)
euc_dist_test = minkowski_distance(locid_pre_test, locid_ori_test, 2)

print('euc and man', euc_dist, man_dist)

def get_accuracies(pre_data, ori_data, prediction_range):
    correct = [0] * prediction_range  # [0, 0, 0, 0, 0]
    for i, prediction in enumerate(pre_data):
        correct_answer = ori_data[i].tolist().index(1)
        best_n = np.sort(prediction)[::-1][:prediction_range]
        # print (i, prediction, best_n, y_test[i], correct_answer)
        for j in range(prediction_range):
            # y_predict = prediction.tolist().index(best_n[j])
            # print (y_predict)
            if prediction.tolist().index(best_n[j]) == correct_answer:
                for k in range(j, prediction_range):
                    correct[k] += 1
    y_accuracies = []

    for i in range(prediction_range):
        print('%s prediction accuracy: %s' % (i + 1, (correct[i] * 1.0) / len(ori_data)))
        y_accuracies.append((correct[i] * 1.0) / len(ori_data))
    print(y_accuracies)

    return y_accuracies

# Predcition Part
print ('*'*80)
print ('Prediction Part')
#print ('Train')
next_predictor_train = get_accuracies(y_pre, y_train, 10)
#print ('*'*20)
#print ('Test')
#next_predictor_test = get_accuracies(y_pre_test, y_test, 10)

# Recognition Part 
print ('*'*80)
print ('Recognition Part')
#print ('Train')
id_recognizer_train = get_accuracies(z_pre, z_train, 10)
#print ('*'*20)
#print ('Test')
#id_recognizer_test = get_accuracies(z_pre_test, z_test, 10)

print ('*'*80)

def plot_training_info(metrics, save, history):

    # summarize history for accuracy
    if 'pre' in metrics:
        
        fig, ax = plt.subplots(figsize=(12,4))
        lns1 = ax.plot(history['acc2'], color='red', label = 'accuracy')
        ax.set_xlabel('epoch', fontsize=14)
        ax.set_ylabel('accuracy', fontsize = 14)
        ax2=ax.twinx()
        lns2 = ax2.plot(history['loss2'], color='blue', label = 'loss')
        ax2.set_ylabel('loss', fontsize = 14)
        ax2.set_yscale("log")
        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='upper left')
        if save == True:
            plt.savefig(PRE_PNG)
            plt.gcf().clear()
        else:
            plt.show()

    # summarize history for loss
    if 'id' in metrics:

        fig, ax = plt.subplots(figsize=(12,4))
        lns1 = ax.plot(history['acc3'], color='red', label = 'accuracy')
        ax.set_xlabel('epoch', fontsize=14)
        ax.set_ylabel('accuracy', fontsize = 14)
        ax2=ax.twinx()
        lns2 = ax2.plot(history['loss3'], color='blue', label = 'loss')
        ax2.set_ylabel('loss', fontsize = 14)
        ax2.set_yscale("log")
        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='upper left')
        if save == True:
            plt.savefig(ID_PNG)
            plt.gcf().clear()
        else:
            plt.show()

    # summarize history for loss
    if 'rec' in metrics:

        fig, ax = plt.subplots(figsize=(12,4))
        lns1 = ax.plot(history['mae1'], color='red', label = 'mae')
        ax.set_xlabel('epoch', fontsize=14)
        ax.set_ylabel('mae', fontsize = 14)
        ax2=ax.twinx()
        lns2 = ax2.plot(history['loss1'], color='blue', label = 'loss')
        ax2.set_ylabel('loss', fontsize = 14)
        ax2.set_yscale("log")
        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='upper left')
        if save == True:
            plt.savefig(REC_PNG)
            plt.gcf().clear()
        else:
            plt.show()

print ('Plotting history...')
plot_training_info(['pre', 'id', 'rec'], True, history)
print ('Plot done...')
print ('Save prediction result...')
with open(PRE_TOPN_RESULT, 'w') as f:
    f.writelines(str(euc_dist))
    f.writelines(str(man_dist))
    f.writelines(str(euc_dist_test))
    f.writelines(str(man_dist_test))
    f.writelines(str(next_predictor_train))
    f.writelines(str(id_recognizer_train))
    f.writelines(str(next_predictor_test))
    f.writelines(str(id_recognizer_test))

print ('Saved Done.')