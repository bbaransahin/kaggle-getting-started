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
	if(len(x.shape) == 1):
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
	elif(len(x.shape) == 2):
		new_matrix = []
		for column_index in range(x.shape[1]):
			min_val = x[0][column_index]
			max_val = x[0][column_index]
			for i in x:
				if(i[column_index] < min_val):
					min_val = i[column_index]
				if(i[column_index] > max_val):
					max_val = i[column_index]
			new_column = []
			for i in x:
				new_column.append(float(i[column_index]-min_val)/float(max_val-min_val))
			new_matrix.append(np.asarray(new_column))
		return concatenate(new_matrix)
	else:
		print("Papatya Normalizer: Given data has to be 1-D or 2-D but given data's dimension is complex for this function.")
		print("Terminating...")
		exit()

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

def nastats(data):
	'''
	This function gives stats of nan values of given data. It just prints it does not return anything.
	Implemented data types:
		numpy.ndarray (with 1-D or 2-D)
	'''
	total_rows_include = 0
	num_columns_include = []
	data_isnan = np.isnan(data)
	for i in range(data_isnan.shape[1]):
		num_columns_include.append(0)
	for i in range(data_isnan.shape[0]):
		if(type(data_isnan[i]) == np.ndarray):
			if(data_isnan[i].any()):
				total_rows_include = total_rows_include+1
			for v in range(data_isnan.shape[1]):
				if(data_isnan[i][v]):
					num_columns_include[v] = num_columns_include[v]+1
		else:
			# TODO: Handle the 1-D numpy.ndarray
			pass
	print("=====================NASTATS=====================")
	print("Total rows include nan:",total_rows_include)
	print("Columns's counts:")
	for i in range(len(num_columns_include)):
		print("\t",i,":",num_columns_include[i])
	print("=================================================")
	print("\n")

def natomedian(data):
	'''
	This function replace nan values with median value.
	'''
	data_isnan = np.isnan(data)
	median_val = np.nanmedian(data)
	print("median val:",median_val)
	for i in range(data_isnan.shape[0]):
		if(type(data_isnan[i]) == np.ndarray):
			for v in range(data_isnan.shape[1]):
				if(data_isnan[i][v]):
					data[i][v] = median_val
		else:
			# TODO: Handle the 1-D numpy.ndarray
			pass
	return data
