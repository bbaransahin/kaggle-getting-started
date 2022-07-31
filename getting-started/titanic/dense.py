'''
To run this program, data files has to be downloaded from kaggle by "kaggle competitions download -c titanic" command and
unzipped to the directory named as "data".
Trained model has accuracy something like 0.86.
'''
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping

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

# Reading data and preview.
print("Reading data files...")
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

print("train data preview:")
print(train.head())
print("\ntest data preview:")
print(test.head())
print("\n")
print("DONE\n")

# Preprocessing data and preview.
print("Preprocessing data...")
train_y = np.asarray(train.pop("Survived"))
train_x = []
train_x.append(onehot(train["Pclass"]))
train_x.append(onehot(train["Sex"]))
train_x.append(normalize(train["Age"]))
train_x.append(train["SibSp"].to_numpy())
train_x.append(train["Parch"].to_numpy())
train_x = concatenate(train_x)
train_x, train_y = removena(train_x, train_y)

test_x = []
test_x.append(onehot(test["Pclass"]))
test_x.append(onehot(test["Sex"]))
test_x.append(normalize(test["Age"]))
test_x.append(test["SibSp"].to_numpy())
test_x.append(test["Parch"].to_numpy())
test_x = concatenate(test_x)

print("train labels:")
print(train_y)
print("train labels shape:", train_y.shape)
print("train features:")
print(train_x)
print("train features shape:", train_x.shape)
print("DONE\n")

# Building model.
print("Building model...")
model = Sequential()
model.add(InputLayer(input_shape=(8,)))
model.add(Dense(5000, kernel_initializer="uniform", activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(1000, kernel_initializer="uniform", activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(10, kernel_initializer="uniform", activation="relu"))
model.add(Dense(1, kernel_initializer="uniform", activation="sigmoid"))
model.summary()
print("DONE\n")

callbacks = [EarlyStopping(patience=3)]

# Compiling model.
print("Compiling model...")
model.compile(
	optimizer="adam",
	loss=BinaryCrossentropy(),
	metrics=["accuracy"]
)
print("DONE\n")

# Training model.
print("Training...")
history = model.fit(
	train_x,
	train_y,
	epochs=17
)
print("DONE\n")

# Plotting history.

# Generating submission file

predictions = model.predict(test_x)
submission = {"PassengerId":[],
        "Survived":[]}
passenger_id = int(test["PassengerId"][0])
for i in predictions:
    submission["PassengerId"].append(passenger_id)
    if(i[0] >= 0.5):
        submission["Survived"].append(1)
    else:
        submission["Survived"].append(0)
    passenger_id+=1
submission = pd.DataFrame.from_dict(submission)
print(submission.head())
submission.to_csv("submission.csv", index=False)
