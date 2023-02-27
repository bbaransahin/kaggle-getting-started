# Data Science London - Scikit Learn

There is not much information about the competition on Kaggle. But it could be used as a simple dataset to learn scikit-learn.

## Solution
First we import the necessary libraries.
```python
import sklearn
import pandas as pd
import numpy as np
```
Pandas and Numpy are used for data reading and processing. And sklearn is the Scikit-Learn library.

```python
rain_x = pd.read_csv('data/train.csv', header=None)
train_y = pd.read_csv('data/trainLabels.csv', header=None)

test = pd.read_csv('data/test.csv', header=None)

print(train_x.head())
print(train_y.head())
print(test.head())

print('train features length =', len(train_x))
print('train labels length =', len(train_y))
print('test length =', len(test))
```
Then it is time to read the csv files with Pandas and printing them to see if we successfully read the data.

```
         0         1         2         3         4         5         6   ...        33        34        35        36        37        38        39
0  0.299403 -1.226624  1.498425 -1.176150  5.289853  0.208297  2.404498  ...  0.293024  3.552681  0.717611  3.305972 -2.715559 -2.682409  0.101050
1 -1.174176  0.332157  0.949919 -1.285328  2.199061 -0.151268 -0.427039  ...  0.468579 -0.517657  0.422326  0.803699  1.213219  1.382932 -1.817761
2  1.192222 -0.414371  0.067054 -2.233568  3.658881  0.089007  0.203439  ...  0.856988 -2.751451 -1.582735  1.672246  0.656438 -0.932473  2.987436
3  1.573270 -0.580318 -0.866332 -0.603812  3.125716  0.870321 -0.161992  ... -1.065252  2.153133  1.563539  2.767117  0.215748  0.619645  1.883397
4 -0.613071 -0.644204  1.112558 -0.032397  3.490142 -0.011935  1.443521  ... -0.205029 -4.744566 -1.520015  1.830651  0.870772 -1.894609  0.408332

[5 rows x 40 columns]
   0
0  1
1  0
2  0
3  1
4  0
         0         1         2         3         4         5         6   ...        33        34        35        36        37        38        39
0  2.808909 -0.242894 -0.546421  0.255162  1.749736 -0.030458 -1.322071  ... -0.479584 -0.244388 -0.672355  0.517860  0.010665 -0.419214  2.818387
1 -0.374101  0.537669  0.081063  0.756773  0.915231  2.557282  3.703187  ... -1.612240  0.179031 -2.924596  0.643610 -1.470939 -0.067408 -0.976265
2 -0.088370  0.154743  0.380716 -1.176126  1.699867 -0.258627 -1.384999  ...  0.483803 -3.542981  0.814561 -1.652948  1.265866 -1.749248  1.773784
3 -0.685635  0.501283  1.873375  0.215224 -3.983468 -0.103637  4.136113  ...  0.285646  2.302069  1.255588 -1.563090 -0.125258 -1.030761 -2.945329
4  0.350867  0.721897 -0.477104 -1.748776 -2.627405  1.075433  4.954253  ...  0.372992  0.450700 -0.211657  1.301359 -0.522164  2.484883  0.039213

[5 rows x 40 columns]
train features length = 1000
train labels length = 1000
test length = 9000
```

We read the data successfully. Now its time to seek if there is any nan value.

```python
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
```

**DataFrame.isna().sum().sum()** returns the number of nan values on a DataFrame object.

The output of the above code segment is:

```
There isn't any nan value on train features
There isn't any nan value on train labels
There isn't any nan value on test dataset
```

Which means our data has no nan value.

```python
mins = train_x.min()
maxs = train_x.max()

print('\tmin\t --- \t max')
for i in range(len(mins)):
    print(mins[i], maxs[i])
```

In the above code we look for maximum and minimum values of columns.

```
	     min	 --- 	 max
-3.365710934320055 3.3262460029909557
-3.4920855905014805 3.583870444370492
-2.6956019378040903 2.546506518106591
-3.460471428852842 3.0887379065173266
-16.421901472901524 17.56534450562528
-3.0412501344576195 3.102997314496187
-7.224760633522439 7.592666376348799
-6.509084243646509 7.130097427981336
-3.1455877547907547 3.1452582309748798
-2.749811733182846 3.919425758354087
-3.304074466459427 3.409653452684431
-3.1574359243847563 3.2530315911997514
-14.706079671024456 12.186445301840688
-3.002151142025112 3.7374225522243134
-6.790633415662062 6.9597363076027285
-2.914728733339179 3.1009351137491263
-3.46404823635581 2.8051965121943114
-2.944093024563838 3.2915436332760755
-8.258306078387369 7.07443237515732
-3.423874744757959 3.343812172447618
-4.251381703455512 2.938032828377816
-2.8226438097572197 3.053261509267568
-6.337521914633045 8.096837537337406
-16.15606959164308 14.37368053850028
-3.2184462050450504 2.9815822578694107
-2.820792134918656 3.662800390973544
-3.0238111279004047 3.2939107503921434
-3.0543836488637304 3.06988491528095
-8.034420590510532 7.4131730292347235
-7.105722603340969 8.812739010572113
-3.379193681730551 2.8447917731511514
-2.9711245367064296 3.6880472499350656
-7.840889691965225 7.160379064069076
-2.9995640785796347 3.3536305306470786
-7.124105345129295 6.005817535530427
-2.952358369706773 3.4205610293237405
-5.452254038673658 6.603499090063512
-3.4739132920764035 3.4925475815377807
-8.05172239306491 5.774119916313516
-7.799086107717904 6.803984337703128
```

Now it is time to preprocess the data. First let's transform dataframes to numpy arrays.

```python
train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
test = np.asarray(test)
```

Then normalize them with builtin numpy function

```python
from sklearn.preprocessing import normalize
train_x = normalize(train_x)
test = normalize(test)
```

I think %10 of validation set is enough for this solution. Let's separate the validation set from training set.

```python
validation_x = train_x[900:]
validation_y = np.ravel(train_y[900:])
train_x = train_x[:900]
train_y = np.ravel(train_y[:900])
```

Printing out the shape and data types of new datasets.

```python
print('validation features shape=', validation_x.shape, '- type:', type(validation_x))
print('validation labels shape=', validation_y.shape, '- type:', type(validation_x))
print('train features shape=', train_x.shape, '- type:', type(validation_x))
print('train labels shape=', train_y.shape, '- type:', type(validation_x))
```

```
validation features shape= (100, 40) - type: <class 'numpy.ndarray'>
validation labels shape= (100,) - type: <class 'numpy.ndarray'>
train features shape= (900, 40) - type: <class 'numpy.ndarray'>
train labels shape= (900,) - type: <class 'numpy.ndarray'>
```

Everything looks fine. It is time to build the model and fine tune it. To do this we can implement a function named as **gradient_boosting()**.

```python
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
```

**GridSearchCV** class is required for fine tuning. It takes **param_grid** as parameter for configurations to try and **cv** as parameter to know how many times to try each configuration. **verbose** determines to output progress or not.

```python
gradient_boosting()
```

Now it is time to run the model.

```
gradient boosting classifier
Fitting 5 folds for each of 36 candidates, totalling 180 fits
[CV] END ....learning_rate=0.1, max_depth=1, n_estimators=50; total time=   0.2s
[CV] END ....learning_rate=0.1, max_depth=1, n_estimators=50; total time=   0.2s
[CV] END ....learning_rate=0.1, max_depth=1, n_estimators=50; total time=   0.2s
[CV] END ....learning_rate=0.1, max_depth=1, n_estimators=50; total time=   0.2s
[CV] END ....learning_rate=0.1, max_depth=1, n_estimators=50; total time=   0.2s
[CV] END ...learning_rate=0.1, max_depth=1, n_estimators=100; total time=   0.5s
[CV] END ...learning_rate=0.1, max_depth=1, n_estimators=100; total time=   0.5s
[CV] END ...learning_rate=0.1, max_depth=1, n_estimators=100; total time=   0.5s
[CV] END ...learning_rate=0.1, max_depth=1, n_estimators=100; total time=   0.5s
[CV] END ...learning_rate=0.1, max_depth=1, n_estimators=100; total time=   0.5s
[CV] END ...learning_rate=0.1, max_depth=1, n_estimators=200; total time=   1.0s
[CV] END ...learning_rate=0.1, max_depth=1, n_estimators=200; total time=   0.9s
[CV] END ...learning_rate=0.1, max_depth=1, n_estimators=200; total time=   0.9s
[CV] END ...learning_rate=0.1, max_depth=1, n_estimators=200; total time=   0.9s
[CV] END ...learning_rate=0.1, max_depth=1, n_estimators=200; total time=   0.9s
[CV] END ....learning_rate=0.1, max_depth=3, n_estimators=50; total time=   0.6s
[CV] END ....learning_rate=0.1, max_depth=3, n_estimators=50; total time=   0.6s
[CV] END ....learning_rate=0.1, max_depth=3, n_estimators=50; total time=   0.6s
[CV] END ....learning_rate=0.1, max_depth=3, n_estimators=50; total time=   0.6s
[CV] END ....learning_rate=0.1, max_depth=3, n_estimators=50; total time=   0.6s
[CV] END ...learning_rate=0.1, max_depth=3, n_estimators=100; total time=   1.2s
[CV] END ...learning_rate=0.1, max_depth=3, n_estimators=100; total time=   1.2s
[CV] END ...learning_rate=0.1, max_depth=3, n_estimators=100; total time=   1.2s
[CV] END ...learning_rate=0.1, max_depth=3, n_estimators=100; total time=   1.2s
[CV] END ...learning_rate=0.1, max_depth=3, n_estimators=100; total time=   1.2s
[CV] END ...learning_rate=0.1, max_depth=3, n_estimators=200; total time=   2.3s
[CV] END ...learning_rate=0.1, max_depth=3, n_estimators=200; total time=   2.3s
[CV] END ...learning_rate=0.1, max_depth=3, n_estimators=200; total time=   2.3s
[CV] END ...learning_rate=0.1, max_depth=3, n_estimators=200; total time=   2.3s
[CV] END ...learning_rate=0.1, max_depth=3, n_estimators=200; total time=   2.3s
[CV] END ....learning_rate=0.1, max_depth=5, n_estimators=50; total time=   0.9s
[CV] END ....learning_rate=0.1, max_depth=5, n_estimators=50; total time=   0.9s
[CV] END ....learning_rate=0.1, max_depth=5, n_estimators=50; total time=   0.9s
[CV] END ....learning_rate=0.1, max_depth=5, n_estimators=50; total time=   0.9s
[CV] END ....learning_rate=0.1, max_depth=5, n_estimators=50; total time=   0.9s
[CV] END ...learning_rate=0.1, max_depth=5, n_estimators=100; total time=   1.8s
[CV] END ...learning_rate=0.1, max_depth=5, n_estimators=100; total time=   1.8s
[CV] END ...learning_rate=0.1, max_depth=5, n_estimators=100; total time=   1.8s
[CV] END ...learning_rate=0.1, max_depth=5, n_estimators=100; total time=   1.9s
[CV] END ...learning_rate=0.1, max_depth=5, n_estimators=100; total time=   1.8s
[CV] END ...learning_rate=0.1, max_depth=5, n_estimators=200; total time=   3.7s
[CV] END ...learning_rate=0.1, max_depth=5, n_estimators=200; total time=   3.7s
[CV] END ...learning_rate=0.1, max_depth=5, n_estimators=200; total time=   3.7s
[CV] END ...learning_rate=0.1, max_depth=5, n_estimators=200; total time=   3.7s
[CV] END ...learning_rate=0.1, max_depth=5, n_estimators=200; total time=   3.7s
[CV] END ....learning_rate=0.1, max_depth=7, n_estimators=50; total time=   1.3s
[CV] END ....learning_rate=0.1, max_depth=7, n_estimators=50; total time=   1.3s
[CV] END ....learning_rate=0.1, max_depth=7, n_estimators=50; total time=   1.3s
[CV] END ....learning_rate=0.1, max_depth=7, n_estimators=50; total time=   1.3s
[CV] END ....learning_rate=0.1, max_depth=7, n_estimators=50; total time=   1.3s
[CV] END ...learning_rate=0.1, max_depth=7, n_estimators=100; total time=   2.6s
[CV] END ...learning_rate=0.1, max_depth=7, n_estimators=100; total time=   2.6s
[CV] END ...learning_rate=0.1, max_depth=7, n_estimators=100; total time=   2.6s
[CV] END ...learning_rate=0.1, max_depth=7, n_estimators=100; total time=   2.6s
[CV] END ...learning_rate=0.1, max_depth=7, n_estimators=100; total time=   2.6s
[CV] END ...learning_rate=0.1, max_depth=7, n_estimators=200; total time=   5.2s
[CV] END ...learning_rate=0.1, max_depth=7, n_estimators=200; total time=   5.2s
[CV] END ...learning_rate=0.1, max_depth=7, n_estimators=200; total time=   5.2s
[CV] END ...learning_rate=0.1, max_depth=7, n_estimators=200; total time=   5.2s
[CV] END ...learning_rate=0.1, max_depth=7, n_estimators=200; total time=   5.3s
[CV] END ...learning_rate=0.05, max_depth=1, n_estimators=50; total time=   0.2s
[CV] END ...learning_rate=0.05, max_depth=1, n_estimators=50; total time=   0.2s
[CV] END ...learning_rate=0.05, max_depth=1, n_estimators=50; total time=   0.2s
[CV] END ...learning_rate=0.05, max_depth=1, n_estimators=50; total time=   0.2s
[CV] END ...learning_rate=0.05, max_depth=1, n_estimators=50; total time=   0.2s
[CV] END ..learning_rate=0.05, max_depth=1, n_estimators=100; total time=   0.5s
[CV] END ..learning_rate=0.05, max_depth=1, n_estimators=100; total time=   0.5s
[CV] END ..learning_rate=0.05, max_depth=1, n_estimators=100; total time=   0.5s
[CV] END ..learning_rate=0.05, max_depth=1, n_estimators=100; total time=   0.5s
[CV] END ..learning_rate=0.05, max_depth=1, n_estimators=100; total time=   0.5s
[CV] END ..learning_rate=0.05, max_depth=1, n_estimators=200; total time=   0.9s
[CV] END ..learning_rate=0.05, max_depth=1, n_estimators=200; total time=   0.9s
[CV] END ..learning_rate=0.05, max_depth=1, n_estimators=200; total time=   0.9s
[CV] END ..learning_rate=0.05, max_depth=1, n_estimators=200; total time=   0.9s
[CV] END ..learning_rate=0.05, max_depth=1, n_estimators=200; total time=   0.9s
[CV] END ...learning_rate=0.05, max_depth=3, n_estimators=50; total time=   0.6s
[CV] END ...learning_rate=0.05, max_depth=3, n_estimators=50; total time=   0.6s
[CV] END ...learning_rate=0.05, max_depth=3, n_estimators=50; total time=   0.6s
[CV] END ...learning_rate=0.05, max_depth=3, n_estimators=50; total time=   0.6s
[CV] END ...learning_rate=0.05, max_depth=3, n_estimators=50; total time=   0.6s
[CV] END ..learning_rate=0.05, max_depth=3, n_estimators=100; total time=   1.1s
[CV] END ..learning_rate=0.05, max_depth=3, n_estimators=100; total time=   1.1s
[CV] END ..learning_rate=0.05, max_depth=3, n_estimators=100; total time=   1.1s
[CV] END ..learning_rate=0.05, max_depth=3, n_estimators=100; total time=   1.1s
[CV] END ..learning_rate=0.05, max_depth=3, n_estimators=100; total time=   1.1s
[CV] END ..learning_rate=0.05, max_depth=3, n_estimators=200; total time=   2.3s
[CV] END ..learning_rate=0.05, max_depth=3, n_estimators=200; total time=   2.3s
[CV] END ..learning_rate=0.05, max_depth=3, n_estimators=200; total time=   2.3s
[CV] END ..learning_rate=0.05, max_depth=3, n_estimators=200; total time=   2.3s
[CV] END ..learning_rate=0.05, max_depth=3, n_estimators=200; total time=   2.3s
[CV] END ...learning_rate=0.05, max_depth=5, n_estimators=50; total time=   0.9s
[CV] END ...learning_rate=0.05, max_depth=5, n_estimators=50; total time=   0.9s
[CV] END ...learning_rate=0.05, max_depth=5, n_estimators=50; total time=   0.9s
[CV] END ...learning_rate=0.05, max_depth=5, n_estimators=50; total time=   0.9s
[CV] END ...learning_rate=0.05, max_depth=5, n_estimators=50; total time=   0.9s
[CV] END ..learning_rate=0.05, max_depth=5, n_estimators=100; total time=   1.8s
[CV] END ..learning_rate=0.05, max_depth=5, n_estimators=100; total time=   1.8s
[CV] END ..learning_rate=0.05, max_depth=5, n_estimators=100; total time=   1.8s
[CV] END ..learning_rate=0.05, max_depth=5, n_estimators=100; total time=   1.8s
[CV] END ..learning_rate=0.05, max_depth=5, n_estimators=100; total time=   1.8s
[CV] END ..learning_rate=0.05, max_depth=5, n_estimators=200; total time=   3.7s
[CV] END ..learning_rate=0.05, max_depth=5, n_estimators=200; total time=   3.7s
[CV] END ..learning_rate=0.05, max_depth=5, n_estimators=200; total time=   3.7s
[CV] END ..learning_rate=0.05, max_depth=5, n_estimators=200; total time=   3.7s
[CV] END ..learning_rate=0.05, max_depth=5, n_estimators=200; total time=   3.7s
[CV] END ...learning_rate=0.05, max_depth=7, n_estimators=50; total time=   1.3s
[CV] END ...learning_rate=0.05, max_depth=7, n_estimators=50; total time=   1.3s
[CV] END ...learning_rate=0.05, max_depth=7, n_estimators=50; total time=   1.3s
[CV] END ...learning_rate=0.05, max_depth=7, n_estimators=50; total time=   1.3s
[CV] END ...learning_rate=0.05, max_depth=7, n_estimators=50; total time=   1.3s
[CV] END ..learning_rate=0.05, max_depth=7, n_estimators=100; total time=   2.6s
[CV] END ..learning_rate=0.05, max_depth=7, n_estimators=100; total time=   2.6s
[CV] END ..learning_rate=0.05, max_depth=7, n_estimators=100; total time=   2.6s
[CV] END ..learning_rate=0.05, max_depth=7, n_estimators=100; total time=   2.6s
[CV] END ..learning_rate=0.05, max_depth=7, n_estimators=100; total time=   2.6s
[CV] END ..learning_rate=0.05, max_depth=7, n_estimators=200; total time=   5.2s
[CV] END ..learning_rate=0.05, max_depth=7, n_estimators=200; total time=   5.3s
[CV] END ..learning_rate=0.05, max_depth=7, n_estimators=200; total time=   5.2s
[CV] END ..learning_rate=0.05, max_depth=7, n_estimators=200; total time=   5.2s
[CV] END ..learning_rate=0.05, max_depth=7, n_estimators=200; total time=   5.3s
[CV] END ..learning_rate=0.001, max_depth=1, n_estimators=50; total time=   0.2s
[CV] END ..learning_rate=0.001, max_depth=1, n_estimators=50; total time=   0.2s
[CV] END ..learning_rate=0.001, max_depth=1, n_estimators=50; total time=   0.2s
[CV] END ..learning_rate=0.001, max_depth=1, n_estimators=50; total time=   0.2s
[CV] END ..learning_rate=0.001, max_depth=1, n_estimators=50; total time=   0.2s
[CV] END .learning_rate=0.001, max_depth=1, n_estimators=100; total time=   0.5s
[CV] END .learning_rate=0.001, max_depth=1, n_estimators=100; total time=   0.5s
[CV] END .learning_rate=0.001, max_depth=1, n_estimators=100; total time=   0.5s
[CV] END .learning_rate=0.001, max_depth=1, n_estimators=100; total time=   0.5s
[CV] END .learning_rate=0.001, max_depth=1, n_estimators=100; total time=   0.5s
[CV] END .learning_rate=0.001, max_depth=1, n_estimators=200; total time=   0.9s
[CV] END .learning_rate=0.001, max_depth=1, n_estimators=200; total time=   0.9s
[CV] END .learning_rate=0.001, max_depth=1, n_estimators=200; total time=   0.9s
[CV] END .learning_rate=0.001, max_depth=1, n_estimators=200; total time=   0.9s
[CV] END .learning_rate=0.001, max_depth=1, n_estimators=200; total time=   0.9s
[CV] END ..learning_rate=0.001, max_depth=3, n_estimators=50; total time=   0.6s
[CV] END ..learning_rate=0.001, max_depth=3, n_estimators=50; total time=   0.6s
[CV] END ..learning_rate=0.001, max_depth=3, n_estimators=50; total time=   0.6s
[CV] END ..learning_rate=0.001, max_depth=3, n_estimators=50; total time=   0.6s
[CV] END ..learning_rate=0.001, max_depth=3, n_estimators=50; total time=   0.6s
[CV] END .learning_rate=0.001, max_depth=3, n_estimators=100; total time=   1.1s
[CV] END .learning_rate=0.001, max_depth=3, n_estimators=100; total time=   1.1s
[CV] END .learning_rate=0.001, max_depth=3, n_estimators=100; total time=   1.1s
[CV] END .learning_rate=0.001, max_depth=3, n_estimators=100; total time=   1.1s
[CV] END .learning_rate=0.001, max_depth=3, n_estimators=100; total time=   1.1s
[CV] END .learning_rate=0.001, max_depth=3, n_estimators=200; total time=   2.2s
[CV] END .learning_rate=0.001, max_depth=3, n_estimators=200; total time=   2.2s
[CV] END .learning_rate=0.001, max_depth=3, n_estimators=200; total time=   2.3s
[CV] END .learning_rate=0.001, max_depth=3, n_estimators=200; total time=   2.3s
[CV] END .learning_rate=0.001, max_depth=3, n_estimators=200; total time=   2.3s
[CV] END ..learning_rate=0.001, max_depth=5, n_estimators=50; total time=   0.9s
[CV] END ..learning_rate=0.001, max_depth=5, n_estimators=50; total time=   0.9s
[CV] END ..learning_rate=0.001, max_depth=5, n_estimators=50; total time=   0.9s
[CV] END ..learning_rate=0.001, max_depth=5, n_estimators=50; total time=   0.9s
[CV] END ..learning_rate=0.001, max_depth=5, n_estimators=50; total time=   0.9s
[CV] END .learning_rate=0.001, max_depth=5, n_estimators=100; total time=   1.8s
[CV] END .learning_rate=0.001, max_depth=5, n_estimators=100; total time=   1.8s
[CV] END .learning_rate=0.001, max_depth=5, n_estimators=100; total time=   1.7s
[CV] END .learning_rate=0.001, max_depth=5, n_estimators=100; total time=   1.8s
[CV] END .learning_rate=0.001, max_depth=5, n_estimators=100; total time=   1.8s
[CV] END .learning_rate=0.001, max_depth=5, n_estimators=200; total time=   3.6s
[CV] END .learning_rate=0.001, max_depth=5, n_estimators=200; total time=   3.6s
[CV] END .learning_rate=0.001, max_depth=5, n_estimators=200; total time=   3.5s
[CV] END .learning_rate=0.001, max_depth=5, n_estimators=200; total time=   3.5s
[CV] END .learning_rate=0.001, max_depth=5, n_estimators=200; total time=   3.6s
[CV] END ..learning_rate=0.001, max_depth=7, n_estimators=50; total time=   1.2s
[CV] END ..learning_rate=0.001, max_depth=7, n_estimators=50; total time=   1.2s
[CV] END ..learning_rate=0.001, max_depth=7, n_estimators=50; total time=   1.1s
[CV] END ..learning_rate=0.001, max_depth=7, n_estimators=50; total time=   1.1s
[CV] END ..learning_rate=0.001, max_depth=7, n_estimators=50; total time=   1.2s
[CV] END .learning_rate=0.001, max_depth=7, n_estimators=100; total time=   2.4s
[CV] END .learning_rate=0.001, max_depth=7, n_estimators=100; total time=   2.4s
[CV] END .learning_rate=0.001, max_depth=7, n_estimators=100; total time=   2.3s
[CV] END .learning_rate=0.001, max_depth=7, n_estimators=100; total time=   2.3s
[CV] END .learning_rate=0.001, max_depth=7, n_estimators=100; total time=   2.4s
[CV] END .learning_rate=0.001, max_depth=7, n_estimators=200; total time=   4.9s
[CV] END .learning_rate=0.001, max_depth=7, n_estimators=200; total time=   4.8s
[CV] END .learning_rate=0.001, max_depth=7, n_estimators=200; total time=   4.5s
[CV] END .learning_rate=0.001, max_depth=7, n_estimators=200; total time=   4.6s
[CV] END .learning_rate=0.001, max_depth=7, n_estimators=200; total time=   4.8s
acc = 0.89
```

The final accuracy that we could get best is 0.89.
