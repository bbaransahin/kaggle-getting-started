'''
To run this program, data files has to be downloaded from kaggle by "kaggle competitions download -c data-science-london-scikit-learn" command and
unzipped to the directory named as "data".
Trained model has accuracy something like 0.87.
'''
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping

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


#Read data files and preview
train_x = pd.read_csv("data/train.csv")
train_y = pd.read_csv("data/trainLabels.csv")

print("Train features:")
print(train_x.head())
print("Train labels:")
print(train_y.head())

#Preprocess data and preview
train_x = train_x.to_numpy()
train_y = train_y.to_numpy()

print("Train features shape:", train_x.shape)
print("Train labels shape:", train_y.shape)

train_x = normalize(train_x)

#Build model
model = Sequential()
model.add(Input(shape=(40,)))
model.add(Dense(1000, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(100, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1, activation="sigmoid"))
model.summary()

#Compile model
model.compile(
	optimizer="adam",
	loss="binary_crossentropy",
	metrics=["accuracy"]
)

#Train
callbacks = []
callbacks.append(EarlyStopping(monitor="loss", patience=4, restore_best_weights=True))
history = model.fit(
	train_x,
	train_y,
	validation_split=0.2,
	callbacks=callbacks,
	epochs=100
)

plothis(history)
