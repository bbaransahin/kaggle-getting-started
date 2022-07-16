'''
To run this program, data files has to be downloaded from kaggle by "kaggle competitions download -c titanic" command and
unzipped to the directory named as "data".
Trained model has accuracy something like 0.86.
'''
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.losses import BinaryCrossentropy

from papatya import onehot
from papatya import concatenate
from papatya import removena
from papatya import normalize
from papatya import plothis

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
model.add(Dense(1500, kernel_initializer="uniform", activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1000, kernel_initializer="uniform", activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(100, kernel_initializer="uniform", activation="relu"))
model.add(Dense(1, kernel_initializer="uniform", activation="sigmoid"))
print(model.summary())
print("DONE\n")

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
	validation_split=0.2,
	epochs=90
)
print("DONE\n")

# Plotting history.
plothis(history)
