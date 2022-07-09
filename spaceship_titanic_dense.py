'''
To run this program, data files has to be downloaded from kaggle by "kaggle competitions download -c spaceship-titanic" command and
unzipped to the directory named as "data".
Trained model has accuracy something like 0.81.
'''
print("Importing libraries...")
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import RandomNormal
print("DONE\n")

# Read data file

print("Reading data and preprocessing...")
train = pd.read_csv("data/train.csv")

train = train.dropna()

print(train.head())

# Preprocess data
def to_onehot(s):
    '''
    This function converts pandas series object to onehot format.
    '''
    vals = []
    indices = []
    for i,v in s.iteritems():
        if(not v in vals):
            vals.append(v)
        indices.append(vals.index(v))

    depth = len(vals)
    return np.asarray(tf.one_hot(indices, depth))

def concatenate(s):
    '''
    This function transforms list of numpy arrays to a numpy array.
    '''
    for i in s:
        print("shp:", i.shape)
    new_data = []
    for i in range(s[0].shape[0]):
        new_item = []
        for j in s:
            if(len(j.shape) == 1):
                new_item.append(j[i])
            elif(len(j.shape) == 2):
                for k in j[i]: # TODO: There is a problem with concatenating cabin data.
                    new_item.append(k)
        new_data.append(new_item)
    return np.asarray(new_data)

def normalize(s):
    max_val = s[0]
    min_val = s[0]
    for i in s:
        if(i > max_val):
            max_val = i
        if(i < min_val):
            min_val = i
    for i in s:
        i = float(i-min_val)/(max_val-min_val)
    return s

x_train = []

x_train.append(to_onehot(train.pop("HomePlanet")))
x_train.append(to_onehot(train.pop("CryoSleep")))
x_train.append(to_onehot(train.pop("Destination")))
x_train.append(normalize(train.pop("Age").to_numpy()))
x_train.append(normalize(train.pop("RoomService").to_numpy()))
x_train.append(normalize(train.pop("FoodCourt").to_numpy()))
x_train.append(normalize(train.pop("ShoppingMall").to_numpy()))
x_train.append(normalize(train.pop("Spa").to_numpy()))
x_train.append(normalize(train.pop("VRDeck").to_numpy()))
x_train.append(to_onehot(train.pop("VIP")))

cabin_data = train.pop("Cabin")
for i,v in cabin_data.iteritems():
    cabin_data.loc[i] = v.split('/')[0]
x_train.append(to_onehot(cabin_data))

x_train = concatenate(x_train)

y_train = to_onehot(train.pop("Transported"))

print("x_train shape:", x_train.shape)
print("x_train:")
print(x_train)
print("y_train.shape:", y_train.shape)
print("y_train:")
print(y_train)
print("DONE\n")

# Build model

print("Building model...")
model = Sequential([
    InputLayer(input_shape=(24,)),
    Dense(500, activation='relu', kernel_initializer=RandomNormal(stddev=0.01)),
    Dropout(0.35),
    Dense(100, activation='relu', kernel_initializer=RandomNormal(stddev=0.01)),
    Dense(2, activation='softmax')
    ])

model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
        )
print("DONE\n")

callback = EarlyStopping(monitor="val_loss", patience=3)

# Train

print("Training...")
history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        callbacks=[callback],
        epochs=250
        )
print("DONE\n")

# Plot results

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('los')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.show()
