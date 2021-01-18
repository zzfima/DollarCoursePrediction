import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

usd_rate = pd.DataFrame(pd.read_excel('RC_F01_01_2017_T01_01_2020.xlsx'))
oil_rate = pd.read_excel('RBRTEd.xls', sheet_name=1, skiprows=2, names=['date', 'oil_price'])

# join 2 tables by date.
# Oil length = 8542, dollar legth = 741, so joined table will be 741
# Put attention - in joined table only 583 oil values - it have NaN
# Why? Because dollar exchange works at [Tu, We, Th, Fr, Sa],
# while oil exchange works at [Tu, We, Th, Fr]
df = usd_rate.set_index('data').join(oil_rate.set_index('date'))

# remove odd columns
df.drop(axis=1, columns=['nominal', 'cdx'], inplace=True)

# replace NaN with ffill - take previous value, put instad of NaN
df.fillna(method='ffill', inplace=True)

# becouse of some joins etc lets resetindex
df.reset_index(inplace=True)

# There no enough columns to make ML regression, so need add additional columns
# Lets add:
# oil prices for past 7 days
# USD prices for past 7 days
# Year, Month, Day of week
df['year'] = df['data'].dt.year
df['month'] = df['data'].dt.month
df['weekday'] = df['data'].dt.weekday

# create columns usd_lag_ and put inside with a shift
# oil, usd
past_days = 7

for day in range(past_days):
    d = day + 1
    df[f"usd_lag_{d}"] = df['curs'].shift(d)
    df[f"oil_lag_{d}"] = df['oil_price'].shift(d)

df['usd_week'] = df['curs'].shift(1).rolling(window=7).median()
df['oil_week'] = df['oil_price'].shift(1).rolling(window=7).median()
final_df = pd.get_dummies(data=df, columns=['year', 'month', 'year']).drop(axis=1, columns=['data', 'oil_price'])
final_df.dropna(inplace=True)
final_df.reset_index(inplace=True)
final_df.drop(axis=1, columns=['index'], inplace=True)
X = final_df.drop(axis=1, columns=['curs'])
y = final_df['curs']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
model = LinearRegression()
model.fit(X_train, y_train)
prediction = model.predict(X_test)
MAE = mean_absolute_error(y_test, prediction)
print(MAE)
