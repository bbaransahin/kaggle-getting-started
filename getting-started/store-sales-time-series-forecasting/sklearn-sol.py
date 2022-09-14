'''
This codes taken from Kaggle
It is designed to work on Kaggle's console ( adding dataset to working dir could be necessary. ).
'''

# Importing Necessary Libraries
from IPython.display import HTML
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from pathlib import Path
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

from learntools.time_series.style import *  # plot style settings
from learntools.time_series.utils import plot_periodogram, seasonal_plot

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error

import pandas as pd
import numpy as np
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Loading Datasets
#-------------------------------------------------------------------------------------------------------------------------
# READING ALL CSV FILES...
#-------------------------------------------------------------------------------------------------------------------------
oil_price = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/oil.csv')
sample_submission = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/sample_submission.csv')
holidays_events = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/holidays_events.csv')
stores = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/stores.csv',)
train = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/train.csv',
                   parse_dates=['date'])
test = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/test.csv')
transactions = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/transactions.csv')

#-------------------------------------------------------------------------------------------------------------------------
#Let's put all datasets in one dictionary. It's going to be helpful while seeking miss information for example.
#-------------------------------------------------------------------------------------------------------------------------
DATASETS = {'oil_price':oil_price,'sample_submission':sample_submission,'holidays_events':holidays_events,
            'stores':stores,'train':train,'test':test,'transactions':transactions}

# Miss Information Analysis
#-------------------------------------------------------------------------------------------------------------------------
# Let's check all datasets and define how much miss values they have.
# We are going to use simple functions here as [.isnull()] and [.sum()].
#-------------------------------------------------------------------------------------------------------------------------

for dataset in DATASETS:
    print('-'*30)
    print(dataset)
    print('-'*30)
    print(DATASETS[dataset].isnull().sum())
    print()

#-------------------------------------------------------------------------------------------------------------------------
# It's noticeable that all dataframes have only one with miss information and as you've already guessed it is oil prices.
# We have 43 null values in a column called [dcoilwtico]. After this comment you can find a visualization of this information.
#-------------------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(25,2))
sns.heatmap(oil_price.isna().transpose())

#-------------------------------------------------------------------------------------------------------------------------
# Let's fill in these valuse with function [.fillna] and put next value in each zero case using [backfill] method.
#-------------------------------------------------------------------------------------------------------------------------
oil_price = oil_price.fillna(method = 'backfill')
oil_price.iloc[[0,1,1216,1217],:]

# Preprocessing
#-------------------------------------------------------------------------------------------------------------------------
#Let's combine all families in one list.
#-------------------------------------------------------------------------------------------------------------------------
FAMILIES =  train.family.unique()

#-------------------------------------------------------------------------------------------------------------------------
#Drawing families seperately it's the best way to show that each of them has unique features.
#-------------------------------------------------------------------------------------------------------------------------
FAMILIES =  train.family.unique()
fig,ax = plt.subplots(nrows = 11,ncols=3,figsize=(25,50))
for i,category in enumerate(FAMILIES):
    train_test = train.loc[(train['family']==category)].copy()
    train_test = train_test.groupby('date').mean()
    ax[i//3,i%3].plot(train_test.index,train_test.sales,linewidth=1/2)
    ax[i//3,i%3].set(title=category)

# Modelling
comp_dir = Path('../input/store-sales-time-series-forecasting')

holidays_events = pd.read_csv(
    comp_dir / "holidays_events.csv",
    dtype={
        'type': 'category',
        'locale': 'category',
        'locale_name': 'category',
        'description': 'category',
        'transferred': 'bool',
    },
    parse_dates=['date'],
    infer_datetime_format=True,
)
holidays_events = holidays_events.set_index('date').to_period('D')

store_sales = pd.read_csv(
    comp_dir / 'train.csv',
    usecols=['store_nbr', 'family', 'date', 'sales'],
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'sales': 'float32',
    },
    parse_dates=['date'],
    infer_datetime_format=True,
)
store_sales['date'] = store_sales.date.dt.to_period('D')
store_sales = store_sales.set_index(['store_nbr', 'family', 'date']).sort_index()
average_sales = (
    store_sales
    .groupby('date').mean()
    .squeeze()
    .loc['2017']
)

y = store_sales.unstack(['store_nbr', 'family']).loc["2017"]

# Create training data
fourier = CalendarFourier(freq='M', order=3)
dp = DeterministicProcess(
    index=y.index,
    constant=True,
    order=2,
    seasonal=True,
    additional_terms=[fourier],
    drop=True,
)
X = dp.in_sample()
X['NewYear'] = (X.index.dayofyear == 1)
model = LinearRegression(fit_intercept=False)
model.fit(X, y)
y_pred = pd.DataFrame(model.predict(X), index=X.index, columns=y.columns)

STORE_NBR = '1'  # 1 - 54
FAMILY = 'BOOKS'
# Uncomment to see a list of product families
# display(store_sales.index.get_level_values('family').unique())

ax = y.loc(axis=1)['sales', STORE_NBR, FAMILY].plot(**plot_params)
ax = y_pred.loc(axis=1)['sales', STORE_NBR, FAMILY].plot(ax=ax)
ax = (y.loc(axis=1)['sales', STORE_NBR, FAMILY]- y_pred.loc(axis=1)['sales', STORE_NBR, FAMILY]).plot(ax=ax)
ax.set_title(f'{FAMILY} Sales at Store {STORE_NBR}');

df_test = pd.read_csv(
    comp_dir / 'test.csv',
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'onpromotion': 'uint32',
    },
    parse_dates=['date'],
    infer_datetime_format=True,
)
df_test['date'] = df_test.date.dt.to_period('D')
df_test = df_test.set_index(['store_nbr', 'family', 'date']).sort_index()

# Create features for test set
X_test = dp.out_of_sample(steps=16)
X_test.index.name = 'date'
X_test['NewYear'] = (X_test.index.dayofyear == 1)

# Submission
y_submit = pd.DataFrame(model.predict(X_test), index=X_test.index, columns=y.columns)
y_submit = y_submit.stack(['store_nbr', 'family'])
y_submit = y_submit.join(df_test.id).reindex(columns=['id', 'sales'])
y_submit.to_csv('submission.csv', index=False)
