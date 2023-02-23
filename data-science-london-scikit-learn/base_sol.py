import sklearn
import pandas as pd
import numpy as np

# Reading the data files and scouting
train_x = pd.read_csv('data/train.csv', header=None)
train_y = pd.read_csv('data/trainLabels.csv', header=None)

test = pd.read_csv('data/test.csv', header=None)

print(train_x.head())
print(train_y.head())
print(test.head())

print('train features length =', len(train_x))
print('train labels length =', len(train_y))
print('test length =', len(test))

train_x_nan_count = train_x.isna().sum().sum()
train_y_nan_count = train_y.isna().sum().sum()
test_nan_count = test.isna().sum().sum()

if (train_x_nan_count == 0):
    print("There isn't any nan value on train features")
else:
    print('Number of nan values on train features is', train_x_nan_count)

if (train_y_nan_count == 0):
    print("There isn't any nan value on train labels")
else:
    print('Number of nan values on train labels is', test_nan_count)

if (test_nan_count == 0):
    print("There isn't any nan value on test dataset")
else:
    print('Number of nan values on test dataset is', test_nan_count)

mins = train_x.min()
maxs = train_x.max()

print('\tmin\t --- \t max')
for i in range(len(mins)):
    print(mins[i], maxs[i])

# Preprocessing

train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
test = np.asarray(test)

from sklearn.preprocessing import normalize
train_x = normalize(train_x)
test = normalize(test)

# Train Validation split
validation_x = train_x[900:]
validation_y = np.ravel(train_y[900:])
train_x = train_x[:900]
train_y = np.ravel(train_y[:900])

# Information about preprocessed data
print('validation features shape=', validation_x.shape, '- type:', type(validation_x))
print('validation labels shape=', validation_y.shape, '- type:', type(validation_x))
print('train features shape=', train_x.shape, '- type:', type(validation_x))
print('train labels shape=', train_y.shape, '- type:', type(validation_x))

# Building models and experimenting
def gradient_boosting():
    print('gradient boosting classifier')
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import GridSearchCV
    param_grid = {
            'learning_rate': [0.1, 0.05, 0.001],
            'max_depth': [1, 3, 5, 7],
            'n_estimators': [50, 100, 200]
            }
    model = GradientBoostingClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=5, verbose=2)
    grid_search.fit(train_x, train_y)
    best_model = grid_search.best_estimator_

    acc = best_model.score(validation_x, validation_y)

    print('acc =', acc)

gradient_boosting()
