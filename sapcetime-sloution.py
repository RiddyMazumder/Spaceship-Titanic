# Generated from: sapcetime-sloution.ipynb
# Converted at: 2026-01-09T19:33:42.734Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # Spaceship Titanic  .....![icons8-spaceship-64.png](attachment:0e75621e-b2d7-4b42-a928-1119c86bbcd4.png)
# >  **Predict which passengers are transported to an alternate dimension**                  
# ## Author: RIDDY MAZUMDER
# ## 🔗 Connect with Me
# > [![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/riddymazumder)
# > [![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/RiddyMazumder)
# > [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/riddy-mazumder-7bab46338/)
# > [![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:riddymazumder1971@gmail.com)
# 
# ## Description 
# **This notebook follows a complete end-to-end data science workflow, from loading data to model evaluation and final submission.**  
# ****Each section is clearly explained and well-structured for learning and presentation.****


# ## 1. Libraries Required
# 
# ****In this section, we import all the necessary Python libraries used throughout the project.****  
# **These include libraries for**:
# - **Data manipulation**  
# - **Visualization** 
# - **Machine learning**


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt ###Data Visualzation
import seaborn as sns###Data Visualzation
from sklearn.impute import KNNImputer
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# ## 2. Load Dataset


df_train = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')
df_test = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')
df_test['Transported'] = False
df = pd.concat([df_train, df_test], sort = False)
df.drop(['Name', 'PassengerId'], axis = 1, inplace = True)
df.head()


# ## 3. Data Exploration & Cleaning
# 
# ## 3.1 Overview
# 
# **Check shape, missing values, data types.**


df.shape[0] == df_train.shape[0] + df_test.shape[0]

df.isna().sum()

# ## 3.2 Visualization


sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()
missing_data = pd.DataFrame({
    'Missing_Values': df.isnull().sum(),
    'Percentage': (df.isnull().sum() / len(df)) * 100,
    'Data_Type': df.dtypes
})
missing_data = missing_data[missing_data['Missing_Values'] > 0]

missing_data = missing_data.sort_values(by='Missing_Values', ascending=False)

print("Missing Data Summary:\n")
display(missing_data)

# ## 3.3 Filling missing values


df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand = True)
df = df.drop(columns = ['Cabin'])
df['Deck'] = df['Deck'].fillna('U')
df['Num'] = df['Num'].fillna(-1)
df['Side'] = df['Side'].fillna('U')

df['Destination'].value_counts()

# # 3.4 Encoding 


df['Deck'] = df['Deck'].map({'G' : 0, 'F' : 1, 'E' : 2, 'D' : 3, 'C' : 4, 'B' : 5, 'A' : 6, 'U' : 7, 'T' : 8})
df['Side'] = df['Side'].map({'U' : -1, 'P' : 1, 'S' : 2})

impute_lis = ['Age', 'VIP', 'Num', 'CryoSleep', 'Side', 'Deck', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
rest = list(set(df.columns) - set(impute_lis))
df_rest = df[rest]
imp = KNNImputer()
df_imputed = imp.fit_transform(df[impute_lis])
df_imputed = pd.DataFrame(df_imputed, columns = impute_lis)
df = pd.concat([df_rest.reset_index(drop = True), df_imputed.reset_index(drop = True)], axis = 1)

df['HomePlanet'] = df['HomePlanet'].fillna('U')
df['Destination'] = df['Destination'].fillna('U')
category_colls = ['HomePlanet', 'Destination']

for col in category_colls:
    df = pd.concat([df, pd.get_dummies(df[col], prefix = col)], axis = 1)



# # 3.5 Remove irrelevant columns


df = df.drop(columns = category_colls)

df.head()

# # 3.6 Feature engineering


#feature engineering
bill_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
df['amt_spent'] = df[bill_cols].sum(axis = 1)
df['std_amt_spent'] = df[bill_cols].std(axis = 1)
df['mean_amt_spent'] = df[bill_cols].mean(axis = 1)

df['3_high_cols'] = df['CryoSleep'] + df['HomePlanet_Europa'] + df['Destination_55 Cancri e']
df['3_low_cols'] = df['mean_amt_spent'] + df['amt_spent'] + df['HomePlanet_Earth']

df.corr()['Transported'].sort_values(ascending = False)

df_train, df_test = df[:df_train.shape[0]], df[df_train.shape[0]:]
df_test = df_test.drop(columns = 'Transported')
df_train.shape, df_test.shape

# ## 4. Model Building
# **Libraries Required**


from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# # 4.1 Split Data


X = df_train.drop(columns = 'Transported')
y = df_train['Transported']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model_1 = LogisticRegression()
model_2 = DecisionTreeClassifier()
model_3 = RandomForestClassifier()
model_4 = XGBClassifier()
model_5 = LGBMClassifier()

X → all features (input variables)
y → target variable (Transported, 0 or 1)
Resulting shapes
X_train → features for training
X_test → features for validation
y_train → target for training
y_test → target for validation

# # 4.2 Train Model,Evaluate Model


model_1.fit(X_train, y_train)
pred = model_1.predict(X_test)
accuracy_score(y_test, pred)

model_2.fit(X_train, y_train)
pred = model_2.predict(X_test)
accuracy_score(y_test, pred)

model_3.fit(X_train, y_train)
pred = model_3.predict(X_test)
accuracy_score(y_test, pred)

model_4.fit(X_train, y_train)  
pred = model_4.predict(X_test)
accuracy_score(y_test, pred)

# ## 5. Model Accuracy_Score
# **Predictions on training data**


from sklearn.model_selection import cross_val_score

scores = cross_val_score(model_4, X_train, y_train, cv=5, scoring='accuracy')
print("CV mean:", scores.mean())

# # 6. Submission File


df_dummy = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')
pred = model_4.predict(df_test)

final = pd.DataFrame()
final['PassengerId'] = df_dummy['PassengerId']
final['Transported'] = pred

final.to_csv('submission.csv', index = False)