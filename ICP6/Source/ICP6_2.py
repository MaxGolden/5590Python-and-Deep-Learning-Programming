"""
2. Create MultipleRegression for the “weatherHistory” datasetEvaluate the modelusing RMSE and R2 score.
https://umkc.box.com/s/2da27yz1gc1txm8x3o7z88btw4jvz6sz
**You need to convert the categorical feature to the numeric using the provided code in the slide
**You need to do the same with the Null values (missing data) in the data set

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('weatherHistory.csv')
dataset.columns = [c.replace(' ', '_') for c in dataset.columns]

# deal with the null values
nulls = pd.DataFrame(dataset.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)

# Wrangling the non-numeric Features
categoricals = dataset.select_dtypes(exclude=[np.number])
# categoricals.info()
print(categoricals.describe())

dataset = dataset.drop(['Apparent_Temperature_(C)'], axis=1)

# Transforming and engineering non-numeric features
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset['enc_Summary'] = le.fit_transform(dataset['Summary'].astype('str'))
dataset['enc_Daily_Summary'] = le.fit_transform(dataset['Daily_Summary'].astype('str'))
dataset['enc_Precip_Type'] = le.fit_transform(dataset['Precip_Type'].astype('str'))

# Correlation
numeric_features = dataset.select_dtypes(include=[np.number])
corr = numeric_features.corr()
print(corr['Temperature_(C)'].sort_values(ascending=False)[:5], '\n')
print(corr['Temperature_(C)'].sort_values(ascending=False)[-5:])

# drop low correlation features
dataset = dataset.drop(['Pressure_(millibars)', 'enc_Precip_Type', 'Loud_Cover'], axis=1)
# 'Humidity'

# build a linear regression model
dataset = dataset.select_dtypes(include=[np.number])
dataset = dataset.rename(columns=lambda x: x.replace('Temperature_(C)', 'Temperature'))

dataset['nplog'] = np.log(dataset.Temperature)
dataset = dataset[dataset['nplog'] > 0]
dataset.info()

y = np.log(dataset.Temperature)
# y = dataset.Temperature

# 'Wind_Speed_(km/h)', 'Wind_Bearing_(degrees)', 'Visibility_(km)', 'enc_Summary',
x = dataset.drop(['Temperature', 'nplog'], axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=.33)
from sklearn import linear_model
lr1 = linear_model.LinearRegression()
model = lr1.fit(X_train, y_train)

# RMSE: Interpreting this value is somewhat more intuitive that the
# r-squared value. The RMSE measures the distance between our predicted
# values and actual values
print('r2 is: ', model.score(X_test, y_test))
prediction = model.predict(X_test)
from sklearn.metrics import mean_squared_error
print('rmse: ', mean_squared_error(y_test, prediction))

actual_values = y_test
plt.scatter(prediction, actual_values, alpha=.75,
            color='b')  # alpha helps to show overlapping data
plt.xlabel('Predicted Temperature')
plt.ylabel('Actual Temperature')
plt.title('Linear Regression Model')
plt.show()
