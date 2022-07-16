import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import MeanSquaredError

# Read data

train = pd.read_csv("data/train.csv")
print(train.head())

'''
Notes:
    useful columns:
    MSSubClass -> onehot
    MSZoning -> onehot
    LotFrontage -> normalize
    LotArea -> normalize
    Street -> onehot
    LotShape -> onehot
    LandContour -> onehot
    Utilities -> onehot
    LotConfig -> onehot

    SalePrice is the target
    
This will take lots of time so I will get data automatically by
determining if it is float or something else and eliminate the columns
that has lotsof NaN values.
'''

def getnaperc(s):
    return float(s.isnull().sum())/float(len(s))

def getdepthperc(s):
    vals = []
    for i in s:
        if(not i in vals):
            vals.append(i)
    return float(len(vals))/float(len(s))

def toonehot(s):
    vals = []
    indices = []
    for i in s:
        if(not i in vals):
            vals.append(i)
        indices.append(vals.index(i))
    return tf.one_hot(indices, len(vals))

def normalize(s):
    max_val = s[0]
    min_val = s[0]
    for i in s:
        if(i > max_val):
            max_val = i
        if(i < min_val):
            min_val = i
    n = []
    for i in s:
        n.append((i-min_val)/float(max_val-min_val))
    return np.asarray(n)

def concatenate(nps):
    for i in nps:
        if(not i.shape[0] == nps[0].shape[0]):
            print("Can't concatenate data!!!")
            exit()
    new_data = []
    for i in tqdm(range(nps[0].shape[0])):
        new_item = []
        for n in nps:
            if(len(n.shape) == 1):
                new_item.append(n[i])
            elif(len(n.shape) == 2):
                for v in n[i]:
                    new_item.append(v)
        new_data.append(new_item)
    return np.asarray(new_data)

for i in train:
    if(getnaperc(train[i]) > 0.17):
        train.pop(i)
train = train.dropna()

y_train = train.pop("SalePrice").to_numpy()

train.pop("Id")

x_train = []

try:
    print("loading features..")
    x_train = np.load("data/features.npy")
    print("DONE\n")
except:
    for i in train:
        if(getnaperc(train[i]) > 0.3):
            continue
        elif(type(train[i][10]) == np.float64 or type(train[i][10]) == np.int64):
            print(i,type(train[i][10]),"\t\tnormalize")
            x_train.append(normalize(train[i]))
        elif(getdepthperc(train[i]) < 0.1):
            print(i,type(train[i][10]),"\t\t\tonehot")
            x_train.append(toonehot(train[i]))
    x_train = concatenate(x_train)

    print("saving train features..")
    np.save("data/features.npy", x_train)
    print("DONE\n")

print("x train shape", x_train.shape)
print("y train shape", y_train.shape)
print(y_train)

# Build the model

model = Sequential([
    InputLayer(input_shape=(264,)),
    Dense(1000, activation="relu"),
    Dropout(0.2),
    Dense(500, activation="relu"),
    Dropout(0.2),
    Dense(500, activation="relu"),
    Dropout(0.2),
    Dense(50, activation="relu"),
    Dense(1, activation="relu")
    ])

model.summary()

model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=[MeanSquaredError()]
        )

history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=7
        )

# Plot results

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('los')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.show()
