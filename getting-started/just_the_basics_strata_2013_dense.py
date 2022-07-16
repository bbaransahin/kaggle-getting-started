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

from papatya import normalize
from papatya import natomedian
from papatya import plothis
from papatya import normalize
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
