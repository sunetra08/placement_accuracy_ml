from sklearn.metrics import r2_score, mean_squared_error
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


df = pd.read_csv(
    'C:\\Users\\sunet\\OneDrive\\Desktop\\ASSIGNMENT\\python project\\placement prediction\\Placement_Data_Full_Class.csv')
df['salary'].fillna(0, inplace=True)

# print(df.columns)
# print(df.info())

# df.groupby('gender').count().plot(kind='bar')
# plt.show()

# df['salary'].value_counts().plot(kind='bar', figsize=(100, 50), color='blue')
# plt.show()


gender = {'M': 0, 'F': 1}
df['gender'] = df['gender'].replace(gender)

status = {'Not Placed': 0, 'Placed': 1}
df['status'] = df['status'].replace(status)

# print(df.info())

x = df[['gender', 'ssc_p', 'hsc_p', 'degree_p',
        'mba_p', 'etest_p']]
y = df[['status', 'salary']]

# scatter plot


# df['average'] = df[['gender', 'ssc_p', 'hsc_p', 'degree_p',
#                     'mba_p', 'etest_p']].mean(axis=1)

# print(df.head())

# plt.scatter(df['average'], df['salary'])
# plt.show()


# training and test data split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42)

linear = LinearRegression()
# linear = LogisticRegression()
linear.fit(x_train, y_train)

prediction = linear.predict(x_test)

df1 = pd.DataFrame({'ACTUAL': y_test.values.flatten(),
                   'prediction': prediction.flatten()})
# print(df1)


r2 = r2_score(y_test, prediction)
print("R-squared score:", r2)


mse = mean_squared_error(y_test, prediction)
print("Mean Squared Error:", mse)
