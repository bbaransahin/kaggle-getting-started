import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

def onehot(data):
	'''
	This function takes data as parameter, it could be array numpy array of pandas series or any iterable array like type.
	Then transforms it to onehot format, it could transform any type of data (string, float, etc.).
	'''
	#Converting data to numpy array.
	if(type(data) is pd.core.series.Series):
		if(data.isnull().values.any()):
			print("There is NaN values on the data that just about to converted onehot.")
			data = data.dropna()
			pritn("Rows that includes NaN values removed.")
		data = data.to_numpy()
	
	#Determining depth and transforming unique values to indices.
	values = []
	print("Converting to onehot...")
	for i in tqdm(range(len(data))):
		if(not data[i] in values):
			values.append(data[i])
		data[i] = values.index(data[i])
	depth = len(values)
	print("DONE")
	
	#By the information generated, converting data to onehot via tf.one_hot and returning it as numpy array.
	return np.asarray(tf.one_hot(data, depth))

def concatenate(nd_arrays):
	'''
	This function takes list of numpy arrays as parameter and concatenates them. Given numpy arrays shape[0] must
	be equal, otherwise terminates the program.
	'''
	#Checking if all of the arrays lenghts are equal.
	length = nd_arrays[0].shape[0]
	for i in nd_arrays:
		if(not i.shape[0] == length):
			print("To concatenate all of the arrays lenghts must be equal.")
			print("Terminating.")
			exit()
	
	#Concatenating arrays
	concatenated = []
	print("Concatenating...")
	for i in tqdm(range(length)):
		item = []
		for array in nd_arrays:
			if(len(array.shape) == 1):
				item.append(array[i])
			else:
				for val in array[i]:
					item.append(val)
		concatenated.append(item)
	print("DONE")

	return np.asarray(concatenated)

def removena(x, y):
	'''
	This function takes features data and labels data as parameter, then removes any row that includes nan value.
	This is useful for removing nan values if you almost finished preprocessing and have your data seperated as features
	labels in numpy array format.
	'''
	isnan_arr = np.isnan(x)
	new_x = []
	new_y = []
	for i in range(isnan_arr.shape[0]):
		if(not isnan_arr[i].any()):
			new_x.append(x[i])
			new_y.append(y[i])

	return np.asarray(new_x), np.asarray(new_y)

def normalize(x):
	'''
	This function takes a numpy array or arraylike element as parameter and normalize its values to range[0,1], and returns
	result as numpy array.
	'''
	min_val = x[0]
	max_val = x[0]
	for i in x:
		if(i < min_val):
			min_val = i
		if(i > max_val):
			max_val = i
	
	new_x = []
	for i in x:
		new_x.append(float(i-min_val)/float(max_val-min_val))
	
	return np.asarray(new_x)

def plothis(history):
# "Accuracy"
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.show()
# "Loss"
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.show()
