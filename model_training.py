import numpy as np
import pandas as pd
import joblib  # For saving/loading models
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data
df = pd.read_excel('data.xlsx')

# Data Preprocessing
X = df.drop('PPV (mm/s)', axis=1)
y = df['PPV (mm/s)']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

# Train the RandomForest model
rfr = RandomForestRegressor(random_state=0)
rfr.fit(X_train, y_train)

# Train the DecisionTree model
dtr = DecisionTreeRegressor(random_state=0)
dtr.fit(X_train, y_train)

# Train the XGBoost model
xgb_model = xgb.XGBRegressor(eval_metric='rmse', use_label_encoder=False, random_state=0)
xgb_model.fit(X_train, y_train)

# Save all three models using joblib
joblib.dump(rfr, 'random_forest_model.pkl')
joblib.dump(dtr, 'decision_tree_model.pkl')
joblib.dump(xgb_model, 'xgboost_model.pkl')

print("All models saved successfully!")
