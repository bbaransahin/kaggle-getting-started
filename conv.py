'''
To run this program, data files has to be downloaded from kaggle by "kaggle competitions download -c digit-recognizer" command and
unzipped to the directory named as "data".
Trained model has accuracy something like 0.96.
'''
print("importing modules...")
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
print("DONE\n")

#Read data and preview
print("reading data files...")
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")
print("DONE\n")

train_data_y = train_data.pop("label")
train_data_x = train_data

print("train data label")
print(train_data_y.head())
print("\ntrain data feature")
print(train_data_x.head())
print("\ntest data")
print(test_data.head())
print("\n")

#Preprocess data and review
print("preprocessing data...")
train_data_y = tf.one_hot(train_data_y, 10)

train_data_y = np.asarray(train_data_y)
train_data_x = np.asarray(train_data_x)
test_data = np.asarray(test_data)

train_data_x = train_data_x.reshape([len(train_data_x),28,28,1])
print("DONE\n")

#Build model
print("building model...")
model = Sequential()
model.add(tf.keras.Input(shape=(28,28,1)))
model.add(tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, activation="softmax"))
print(model.summary())
print("DONE\n")

#Compile model
print("compiling model...")
model.compile(
	optimizer="adam",
	loss="categorical_crossentropy",
	metrics=["accuracy"]
)
print("DONE\n")

#Train model
print("training model...")
model.fit(
	train_data_x,
	train_data_y,
	validation_split=0.2,
	epochs=2
)
print("DONE\n")
