from __future__ import absolute_import
from __future__ import division
from __future__	import print_function

import numpy as np
import argparse, os
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import *
import matplotlib.pyplot as plt
#from tensorflow.python.ops import init_ops
from tensorflow.contrib.rnn import BasicLSTMCell, BasicRNNCell, GRUCell
import pandas as pd
import re

skipAmount = 10.0 #This is how frequent the data should be sampled.
emaPredictInFuture = 100 #This is how much into the future it should have to predict
seq_len = 50 #Sequence length.
smoothedPrice = 200 #This just smooths the input. 
  
#THis needs to take in the data, then return the data in a list of
#The output should be in a np array form. Note that the y value doesn't have to be returned. 

def file_data(filename="data/06_01_10.csv"):
	#filename="data/06_01_10.csv"
	myVals = pd.DataFrame.from_csv(filename)
	#myVals['seconds'] = (myVals['seconds']*100.0).astype(int)
	print("MY vals: " , myVals)
	emaInFuture = emaPredictInFuture
	myVals['volume'] = pd.rolling_mean(myVals['volume'],skipAmount).fillna(0)
	#Skipping each 5th val
	myVals = myVals.iloc[::skipAmount, :]
	print("Skipping values: " , myVals)
	myVals['emaPrice'] = pd.rolling_mean(myVals['intPrice'].shift(10),emaInFuture).fillna(0)
	myVals['goodBuy'] = myVals['intPrice']<myVals['emaPrice']
	#myVals['intPrice'] = myVals['intPrice']-myVals['intPrice'].shift(1).fillna(0)
	print("Int price: " , myVals)
	myVals['normPrice'] = myVals['intPrice'] - myVals['intPrice'].mean()
	myVals['stdPrice'] = (myVals['normPrice'])/myVals['normPrice'].std()*1.0
	myVals['volume'] = myVals['volume']-myVals['volume'].shift(1).fillna(0)
	myVals['normVolume'] = myVals['volume'] - myVals['volume'].mean()
	myVals['stdVolume'] = (myVals['normVolume'])/myVals['normVolume'].std()*1.0
	standardDev = myVals['normPrice'].std()*1.0
	meanVal = myVals['intPrice'].mean()
	#Now every 4th row. df.iloc[::5, :]
	#Okay so here I pick the indicators. Let's first do a price EMA
	#Mean of the last 10 values
	#myVals['stdPrice'] = pd.rolling_mean(myVals['stdPrice'],10).fillna(0)

	#myVals['volume'] = pd.rolling_mean(myVals['volume'],20).fillna(0)
	#myVals['volume'] = pd.rolling_mean(myVals['volume'],20).fillna(0)

	print ("standardDev: " , standardDev)
	print ("Mean val: " , meanVal)



	newmyVals = myVals[['intPrice','volume','goodBuy']]

	#newmyVals['intPrice'] = pd.rolling_mean(newmyVals['smoothedPrice'],200).fillna(0)

	meanBuy = myVals['goodBuy'].mean()
	vals = (myVals['goodBuy']-meanBuy)*(myVals['goodBuy']-meanBuy)
	print("Baseline: " , vals.mean()*seq_len)
	allData = newmyVals.as_matrix().astype(float)[1:,:]
	#plt.plot(allData)
	#plt.show()

	print("MYData: " , allData)
	return allData , standardDev , meanVal


def main():
	# --- Set data params ----------------
	#Create Data
	max_len_data = 1000000000

	data , standardDev,meanVal= file_data()
	data2 , standardDev,meanVal= file_data(filename="data/06_11_12.csv")

	n_input = len(data[0])

	n_output = 2
	n_hidden = 125
	learning_rate = 0.001
	decay = 0.9
	numEpochs = 2000
	reuse = False

	#Structure of this will be [weekday,seconds*1000,intPrice,volume]

	X = tf.placeholder("float32",[None,seq_len,2])
	Y = tf.placeholder("int64",[None,seq_len])

	# Input to hidden layer
	cell = None
	h = None
	num_layers = 3
	#h_b = None
	sequence_length = [seq_len] * 1


	cell = BasicLSTMCell(n_hidden, state_is_tuple=True, forget_bias=1)

	cells = core_rnn_cell_impl.MultiRNNCell([BasicLSTMCell(n_hidden, state_is_tuple=True, forget_bias =1) for _ in range(num_layers)],state_is_tuple=True)

	#cells = core_rnn_cell_impl.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
	if h == None:
		h = cells.zero_state(1,tf.float32)

	hidden_out, states = tf.nn.dynamic_rnn(cells, X, sequence_length=sequence_length, dtype=tf.float32,initial_state=h)


	# if h == None:
	# 	h = cell.zero_state(1,tf.int32)
	# hidden_out, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)


	# Hidden Layer to Output
	V_init_val = np.sqrt(6.)/np.sqrt(n_output + n_input)

	V_weights = tf.get_variable("V_weights", shape = [n_hidden, n_output], \
			dtype=tf.float32, initializer=tf.random_uniform_initializer(-V_init_val, V_init_val))
	V_bias = tf.get_variable("V_bias", shape=[n_output], \
			dtype=tf.float32, initializer=tf.constant_initializer(0.01))

	hidden_out_list = tf.unstack(hidden_out, axis=1)
	temp_out = tf.stack([tf.matmul(i, V_weights) for i in hidden_out_list])
	output_data = tf.nn.bias_add(tf.transpose(temp_out, [1,0,2]), V_bias)


	# define evaluate process
	print("Output data: " , output_data)
	print("Labels: " , Y)

	cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_data, labels=Y))
	#correct_pred = tf.equal(tf.round(output_data*standardDev+meanVal), tf.round(Y*standardDev+meanVal))
	#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


	# --- Initialization ----------------------
	global_step = tf.Variable(0, trainable=False)

	learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                           1000, 0.9, staircase=True)
	optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=decay).minimize(cost)
	init = tf.global_variables_initializer()

	for i in tf.global_variables():
		print(i.name)

	def do_validation(f2,data,curEpoch):
		maxIter = int((len(data)/seq_len)-1.0)
		val_losses = []
		training_state = None

		for i in xrange(1,maxIter):
			#Batch sizes of 30. 
			myTrain_x = data[seq_len*i:seq_len*(i+1),0:2].reshape((1,seq_len,2))
			myTrain_y = data[seq_len*i+1:seq_len*(i+1)+1,2:3].reshape((1,seq_len))
			myfeed_dict={X: myTrain_x, Y: myTrain_y}
			if training_state is not None:
				myfeed_dict[h] = training_state
			
			loss,training_state,output_data_2 = sess.run([cost, states,output_data], feed_dict = myfeed_dict)
			val_losses.append(loss)

		valLoss = sum(val_losses)/len(val_losses)

		print("Our File Validation Loss= " + \
				  "{:.6f}".format(valLoss))

		maxIter = int((len(data2)/seq_len)-1.0)
		val_losses = []
		training_state = None

		for i in xrange(1,maxIter):
			#Batch sizes of 30. 
			myTrain_x = data2[seq_len*i:seq_len*(i+1),0:2].reshape((1,seq_len,2))
			myTrain_y = data2[seq_len*i+1:seq_len*(i+1)+1,2:3].reshape((1,seq_len))
			myfeed_dict={X: myTrain_x, Y: myTrain_y}
			if training_state is not None:
				myfeed_dict[h] = training_state
			
			loss,training_state,output_data_2 = sess.run([cost, states,output_data], feed_dict = myfeed_dict)
			val_losses.append(loss)
			
		valLoss = sum(val_losses)/len(val_losses)

		print("Real Validation Loss= " + \
				  "{:.6f}".format(valLoss))
		f2.write("%d\t%f\n"%(curEpoch, valLoss))
		f2.flush()

	step = 0
	savename="modelFile"
	saver = tf.train.Saver()
	numFile = 0

	while True:
		val_loss_file = "results/4VolumeTest/"+"val_loss" + str(numFile) + ".txt"
		if os.path.isfile(val_loss_file):
			numFile += 1
		else:
			break
	f2 = open(val_loss_file,'w')


	with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=False)) as sess:
		print("Session Created")
		sess.run(init)
		if reuse:
			new_saver = tf.train.import_meta_graph("modelFile"+'.meta')
			new_saver.restore(sess, tf.train.latest_checkpoint('./'))

		steps = []
		losses = []
		accs = []
		validation_losses = []
		curEpoch = 0

		
		training_state = None
		i = 0
		print ("Number train: " , len(data))
		train_file_name = "loss.csv"
		train_loss_file = open(train_file_name,'w')
		

		while curEpoch < numEpochs:
			i += 1
			#Batch sizes of 30
			if ((i+1) > (len(data)-1.0)/seq_len):
				i = 1
				curEpoch += 1
			myTrain_x = data[seq_len*i:seq_len*(i+1),0:2].reshape((1,seq_len,2))
			myTrain_y = data[seq_len*i+1:seq_len*(i+1)+1,2:3].reshape((1,seq_len))

			myfeed_dict={X: myTrain_x, Y: myTrain_y}
			if training_state is not None:
				myfeed_dict[h] = training_state
			
			empty,loss,training_state,output_data_2 = sess.run([optimizer,  cost, states,output_data], feed_dict = myfeed_dict)
			#print("TrainX: ", myTrain_x)
			
			print("Epoch: " + str(curEpoch) + " Iter " + str(i) + ", Minibatch Loss= " + \
				  "{:.6f}".format(loss))
			if (curEpoch % 5 == 0 and i ==1):
				outputVal = np.array(output_data_2*1.0+0.0)
				correctVal = myTrain_y
				#outputVal = np.array(output_data_2*standardDev+meanVal)
				#correctVal = myTrain_y*standardDev+meanVal
				print("Output: " , outputVal)
				print("My train: " , correctVal)
				#print("Output - myTrain: " , outputVal-correctVal)
				print("My loss: " , loss)
				
				train_loss_file.write(str(loss*standardDev)+"\n")
				train_loss_file.flush()
				do_validation(f2,data,curEpoch)
				saver.save(sess,savename)

		train_loss_file.close()
			

			
		print("Optimization Finished!")
		


if __name__=="__main__":
	
	main()
