# -*- coding: utf-8 -*-
"""regression model.ipynb

Original file is located at
    https://colab.research.google.com/drive/1BTRrB5pyH6Fuqt6Dwpq6wZkNEdZuWGbI
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

# %matplotlib inline
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

StartupDF = pd.read_csv("/content/final_filled_startup_data.csv")

StartupDF.head()

StartupDF.tail()

len(StartupDF)

import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

StartupDF.info()

StartupDF.describe()

StartupDF.columns

sns.heatmap(StartupDF.corr(), annot=True)

"""### **Train And Test Split**"""

X = StartupDF[['state_code', 'latitude', 'longitude', 'zip_code', 'id', 'labels',
       'founded_at', 'closed_at', 'first_funding_at', 'last_funding_at',
       'age_first_funding_year', 'age_last_funding_year',
       'age_first_milestone_year', 'age_last_milestone_year', 'relationships',
       'funding_rounds', 'funding_total_usd', 'milestones', 'state_code.1',
       'is_1', 'is_3', 'is_2', 'is_4', 'is_otherstate', '1tegory_code',
       'is_software', 'is_web', 'is_mobile', 'is_enterprise', 'is_advertising',
       'is_gamesvideo', 'is_ecommerce', 'is_biotech', 'is_consulting',
       'is_other1tegory', 'object_id', 'has_VC', 'has_angel', 'has_roundA',
       'has_roundB', 'has_roundC', 'has_roundD', 'avg_participants',
       'is_top500']]

y = StartupDF['status']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 0.40, random_state=10)

X_train

"""# Create Linear Regression Model"""

lm = LinearRegression()

lm.fit(X_train, y_train)

coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['status'])
coeff_df

prediction = lm.predict(X_test)

plt.scatter(y_test, prediction)

sns.distplot((y_test-prediction), bins=50)

"""## Test Data"""

TestDF = pd.read_csv("/content/final_filled_startup_data.csv")

TestDF.head()

len(TestDF)

TestDF = pd.read_csv("/content/final_filled_startup_data.csv")
TestDF = TestDF.drop(columns=['status'])
test_prediction = lm.predict(TestDF.iloc[[0]])

print("Prediction for the first row of the new data:", test_prediction)
if(test_prediction >=0.5):
  print("This startup will achieve success and secure funding from investors.")
else:
  print("No, this startup will not succeed in securing funding from investors.")

TestDF1 = pd.read_csv("/content/final_filled_startup_data.csv")
TestDF1 = TestDF1.drop(columns=['status'])
test_prediction = lm.predict(TestDF1.iloc[[5]])

print("Prediction for the first row of the new data:", test_prediction)
if(test_prediction >=0.1):
  print("This startup will achieve success and secure funding from investors.")
else:
  print("No, this startup will not succeed in securing funding from investors.")

print("Mean squared error:", mean_squared_error(y_test, prediction))

"""- MSE < 1e-3: This is considered a very good MSE, indicating a very small prediction error.

- 1e-3 < MSE < 1: This is a good MSE, indicating a small prediction error.

- 1 < MSE < 10: This is an acceptable MSE, but it may indicate that the model could be improved.

- MSE > 10: This is a bad MSE, indicating a large prediction error.

The end
"""
