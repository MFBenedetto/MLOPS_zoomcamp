import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

#%%
df = pd.read_parquet('fhv_tripdata_2021-01.parquet')
df_val = pd.read_parquet('fhv_tripdata_2021-02.parquet')

#%%
df['duration'] = df.dropOff_datetime - df.pickup_datetime
df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
df.duration.mean()
print(f'N rows before {df.shape[0]}')

df = df[(df.duration >= 1) & (df.duration <= 60)]
print(f'N rows after {df.shape[0]}')

print(df['PUlocationID'].isna().sum()/df.shape[0])
df['PUlocationID'].fillna(-1, inplace=True)
df['DOlocationID'].fillna(-1, inplace=True)

categorical = ['PUlocationID', 'DOlocationID']

df[categorical] = df[categorical].astype(str)

#%%
train_dicts = df[categorical].to_dict(orient='records')

dv = DictVectorizer()
X_train = dv.fit_transform(train_dicts)

#%%
target = 'duration'
y_train = df[target].values

#%%
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_train)

RMSE_train = mean_squared_error(y_train, y_pred, squared=False)
print(f'RMSE train {RMSE_train}')
#%%
df_val['duration'] = df_val.dropOff_datetime - df_val.pickup_datetime
df_val.duration = df_val.duration.apply(lambda td: td.total_seconds() / 60)

df_val = df_val[(df_val.duration >= 1) & (df_val.duration <= 60)]

val_dicts = df_val[categorical].to_dict(orient='records')
X_val = dv.transform(val_dicts)
y_val = df_val[target].values

RMSE_val = mean_squared_error(y_val,lr.predict(X_val), squared=False)
print(f'RMSE val {RMSE_val}')
