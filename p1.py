import pickle
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df = pd.read_csv(
    'C:\\Users\\sunet\\OneDrive\\Desktop\\ASSIGNMENT\\python project\\PLACEMENT_ACCURACY_ML\\Placement_Data_Full_Class.csv')
df['salary'].fillna(0, inplace=True)

gender = {'M': 0, 'F': 1}
df['gender'] = df['gender'].replace(gender)

df['gender'] = df['gender'].astype(int)

# status = {'Not Placed': 0, 'Placed': 1}
# df['status'] = df['status'].replace(status)

x = df[['gender', 'ssc_p', 'hsc_p', 'degree_p', 'mba_p', 'etest_p']]
y = df['status']

# training and test data split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42)

logistic = LogisticRegression()
logistic.fit(x_train, y_train)

prediction = logistic.predict(x_test)

df1 = pd.DataFrame({'ACTUAL': y_test.values.flatten(),
                   'prediction': prediction.flatten()})

# print(df1.head())

accuracy = accuracy_score(y_test, prediction)
print("Accuracy:", accuracy)


gender_input = int(
    input("Enter gender (for male enter 0 for female enter 1): "))
ssc_input = float(input("Enter SSC percentage: "))
hsc_input = float(input("Enter HSC percentage: "))
degree_input = float(input("Enter degree percentage: "))
mba_input = float(input("Enter MBA percentage: "))
etest_input = float(input("Enter employment test percentage: "))

# Create a new data point using the user input
new_data = pd.DataFrame({'gender': [gender_input],
                         'ssc_p': [ssc_input],
                         'hsc_p': [hsc_input],
                         'degree_p': [degree_input],
                         'mba_p': [mba_input],
                         'etest_p': [etest_input]})


new_prediction = logistic.predict(new_data)

print("Prediction for the new data point:", new_prediction)

with open('placement_prediction.pkl', 'wb') as file:
    pickle.dump(logistic, file)
