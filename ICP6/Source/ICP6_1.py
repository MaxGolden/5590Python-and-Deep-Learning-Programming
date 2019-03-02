"""
1. Delete all the anomaly data for the GarageArea field(for the same data set in the
use case: House Prices).
* for this task you need to plot GaurageArea field and SalePrice in scatter plot,
then check which numbers are anomaly.

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# plot GaurageArea field and SalePrice in scatter plot
plt.scatter(x=train['GarageArea'], y=np.log(train.SalePrice))
plt.ylabel('Sale Price')
plt.xlabel('Above grade (ground) living area square feet')
plt.show()

# We will create a new dataframe with some outliers removed.
train = train[train['GarageArea'] < 1200]
# train = train[train['GarageArea'] > 0]

plt.scatter(x=train['GarageArea'], y=np.log(train.SalePrice))
plt.xlim(-200, 1600)  # This forces the same scale as before
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()














