'''
To run this program, data files has to be downloaded from kaggle by "kaggle competitions download -c just-the-basics-strata-2013" command and
unzipped to the directory named as "data".
Trained model has accuracy something like 0.80.
'''
print("Importing modules...")
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import Input

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

print("DONE\n")

# Read data and preview
print("Reading data files...")
train_x = pd.read_csv("data/train.csv")
train_y = pd.read_csv("data/train_labels.csv")
print(train_x.head())
print(train_y.head())
print("DONE\n")

# Preprocess data and preview
print("Preprocessing data...")
train_x = train_x.to_numpy()
train_y = train_y.to_numpy()

train_x = natomedian(train_x)
train_x = normalize(train_x)

print("Train features:")
print("Shape:", train_x.shape)
print("Train labels:")
print("Shape:", train_y.shape)
print("DONE\n")

#Build the model
model = Sequential()
model.add(Input(shape=(100,)))
model.add(Dense(250, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(500, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(250, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(125, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(25, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(1, activation="sigmoid"))
model.summary()

#Compile the model
model.compile(
	optimizer="adam",
	loss="binary_crossentropy",
	metrics=["accuracy"]
)

#Train
history = model.fit(
	train_x,
	train_y,
	validation_split=0.2,
	epochs=25
)

plothis(history)
