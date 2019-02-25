import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

# load data set
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
# combine = [train_df, test_df]


# 1. Find the correlation between Survived (target column) and Sex column. Do you think we should keep it?
# I observe significant correlation (>0.5) among Pclass=1 and Survived, Sex = 'female' and Survived.
# So, I decide to include this feature in our model.
Pcl_corr = train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
Sex_corr = train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
Sib_corr = train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
par_corr = train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Correcting by dropping features
print("The correlation between Survived and Sex is \n ", Sex_corr)


train_df = train_df.drop(['Ticket', 'Cabin', 'Parch', 'SibSp', 'Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin', 'Parch', 'SibSp', 'Name', 'PassengerId'], axis=1)

# Converting a categorical feature
train_df['Sex'] = train_df['Sex'].map({'female': 1, 'male': 0}).astype(int)
test_df['Sex'] = test_df['Sex'].map({'female': 1, 'male': 0}).astype(int)

g = sns.FacetGrid(train_df, col='Pclass')
g.map(plt.hist, 'Age', bins=20)
plt.show()
g = sns.FacetGrid(train_df, col='Sex')
g.map(plt.hist, 'Age', bins=20)
plt.show()
g = sns.FacetGrid(train_df, col='Embarked')
g.map(plt.hist, 'Age', bins=20)
plt.show()
g_fare = sns.FacetGrid(train_df, hue='Age', size=4)
g_fare.map(plt.scatter, 'Age', "Fare")
plt.show()

freq_port = train_df.Embarked.dropna().mode()[0]
meanAge = int(train_df.Age.dropna().mean())

mean_p1_s1 = int(train_df[(train_df.Pclass == 1) & (train_df.Sex == 1)].Age.dropna().mean())
mean_p3_s1 = int(train_df[(train_df.Pclass == 3) & (train_df.Sex == 1)].Age.dropna().mean())
mean_p1_s0 = int(train_df[(train_df.Pclass == 1) & (train_df.Sex == 0)].Age.dropna().mean())
mean_p3_s0 = int(train_df[(train_df.Pclass == 3) & (train_df.Sex == 0)].Age.dropna().mean())
print(mean_p1_s1, mean_p3_s1, mean_p1_s0, mean_p3_s0)

for index, row in train_df.iterrows():
    if math.isnan(row['Age']):
        if row.Pclass == 1:
            if row.Sex == 1:
                train_df.at[index, 'Age'] = mean_p1_s1
            elif row.Sex == 0:
                train_df.at[index, 'Age'] = mean_p1_s0

        elif row.Pclass == 3:
            if row.Sex == 1:
                train_df.at[index, 'Age'] = mean_p3_s1
            elif row.Sex == 0:
                train_df.at[index, 'Age'] = mean_p3_s0
        else:
            train_df.at[index, 'Age'] = meanAge

# for index, row in test_df.iterrows():
#     if math.isnan(row['Age']):
#         if row.Pclass == 1:
#             if row.Sex == 1:
#                 test_df.at[index, 'Age'] = mean_p1_s1
#             elif row.Sex == 0:
#                 test_df.at[index, 'Age'] = mean_p1_s0
#
#         elif row.Pclass == 3:
#             if row.Sex == 1:
#                 test_df.at[index, 'Age'] = mean_p3_s1
#             elif row.Sex == 0:
#                 test_df.at[index, 'Age'] = mean_p3_s0
#         else:
#             test_df.at[index, 'Age'] = meanAge
#
# train_df['Age'] = train_df['Age'].fillna(meanAge)
# test_df['Age'] = test_df['Age'].fillna(meanAge)

train_df['Embarked'] = train_df['Embarked'].fillna(freq_port)
train_df['Embarked'] = train_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

test_df['Embarked'] = test_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)


train_df.to_csv('test_preprocessed_max_train.csv', index=False)
fare_mean = int(train_df.Fare.dropna().mean())
test_df['Fare'] = test_df['Fare'].fillna(fare_mean)
test_df.to_csv('test_preprocessed_max_test.csv', index=False)
