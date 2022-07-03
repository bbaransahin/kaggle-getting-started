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

from papatya import normalize
from papatya import plothis

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
