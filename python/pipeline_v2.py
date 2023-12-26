import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import matplotlib.pyplot as plt

normal_df = pd.read_csv('../data/alfa-romeo-normal-data.csv')
normal_df.head()

# Standardizing the data
scaler = StandardScaler()
scaled_normal_df = scaler.fit_transform(normal_df)

# Train Isolation Forest on normal data
iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
iso_forest.fit(scaled_normal_df)


# Save the scaler and model to disk
dump(scaler, '../models/scaler.joblib')
dump(iso_forest, '../models/iforest.joblib')

# Creating the DataFrame
faulty_df = pd.read_csv('../data/alfa-romeo-faulty-data.csv')
faulty_df.head()

# Load the scaler and model from disk
scaler = load('../models/scaler.joblib')
model = load('../models/iforest.joblib')

# Convert the 'Datetime' column to datetime
faulty_df['Datetime'] = pd.to_datetime(faulty_df['Datetime'])

# Store the 'Datetime' column in a separate variable before dropping it
datetime_col = faulty_df['Datetime']
faulty_df = faulty_df.drop(columns=['Datetime'])

scaled_faulty_df = scaler.fit_transform(faulty_df)

# Apply the model to the new data to predict anomalies
anomaly_scores = iso_forest.decision_function(scaled_faulty_df)
anomaly_labels = iso_forest.predict(scaled_faulty_df)

# Add a column to the faulty data to show anomalies
faulty_df['Anomaly_Score'] = anomaly_scores
faulty_df['Anomaly_Score_IFR_Norm'] = faulty_df['Anomaly_Score'] * 1000
faulty_df['Anomaly_Score_IPW_Norm'] = faulty_df['Anomaly_Score'] * 10
faulty_df['Anomaly_Score_IT_Norm'] = faulty_df['Anomaly_Score'] * 100
faulty_df['Anomaly_Label'] = anomaly_labels

# Add the 'Datetime' column back to the dataframe
faulty_df['Datetime'] = datetime_col

faulty_df.to_csv('../data/alfa-romeo-faulty-data-anomalies.csv', index=False)