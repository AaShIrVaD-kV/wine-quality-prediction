import pandas as pd 
import numpy as np
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

df = pd.read_csv(r"C:\Users\HP\OneDrive\Documents\PROJECTS\DATA SCIENCE PROJECT\WINE QUALITY\winequality-red.csv", sep=';')

# Features and Target
X = df.drop('quality', axis=1)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- Linear Regression ----------------
model1 = LinearRegression()
model1.fit(X_train, y_train)

y_pred = model1.predict(X_test)

print("Linear Regression Results:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


# ---------------- Random Forest ----------------
model2 = RandomForestRegressor()
model2.fit(X_train, y_train)

y_pred = model2.predict(X_test)

print("\nRandom Forest Results:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# SAVE THE MODEL
with open('model.pkl', 'wb') as file:
    pkl.dump(model2, file)


# ---------------- AdaBoost ----------------
model3 = AdaBoostRegressor()
model3.fit(X_train, y_train)

y_pred = model3.predict(X_test)

print("\nAdaBoost Results:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


# ---------------- Gradient Boosting ----------------
model4 = GradientBoostingRegressor()
model4.fit(X_train, y_train)

y_pred = model4.predict(X_test)

print("\nGradient Boosting Results:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

#evaluvating standard scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model5 = RandomForestRegressor()
model5.fit(X_train_scaled, y_train)
y_pred = model5.predict(X_test_scaled)
print("\nRandom Forest with Standard Scaler Results:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

#creating model pkl for standard scaler
with open('model_scaled.pkl', 'wb') as file:
    pkl.dump(model5, file)