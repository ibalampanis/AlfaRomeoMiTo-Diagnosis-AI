# Harnessing Data and AI to Diagnose Car Faults: A Data Analyst’s Journey with an Alfa Romeo MiTo

![](https://miro.medium.com/v2/resize:fit:700/1*dHdAChal1T4W5bQaxf804g.png)

## Introduction

As a Data Analyst and a car enthusiast, I embarked on a unique journey, blending my professional skills with my passion for automobiles. This story is about how I used data logging and machine learning to identify a potential fault in my 2014 Alfa Romeo MiTo 1.3 JTDm.

## OBD-2 Adapter: The key factor of our mission

The On-Board Diagnostics (OBD-II) adapter is not just a tool but a gateway to understanding the intricate details of my car’s functioning. It provides real-time access to a car’s status and performance data, including engine performance to intricate fuel injection details. The adapter that I used is described [here](https://www.amazon.com/Thinkdiag-Bluetooth-Bidirectional-Diagnostic-Functions/dp/B08YWTJY4F?th=1).

## The Data-Driven Approach

The process began with equipping my Alfa Romeo with the OBD-2 adapter, transforming every journey into a data collection mission. This setup allowed me to monitor various aspects of the car’s performance in real-time.

## Navigating Through the Sea of Data

The challenge was to analyze the extensive data and pinpoint any anomalies indicative of potential issues. This led me to employ the Isolation Forest algorithm, an effective tool in the realm of anomaly detection.

## A Deep Dive into the Technical Process - Data Collection and Preparation

Python Snippet 0: Importing the Python packages

```python
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import matplotlib.pyplot as plt
```

_Explanation:_

- `pandas`: This is a powerful data manipulation library in Python. It provides data structures and functions needed to manipulate structured data.
- `sklearn.ensemble.IsolationForest`: This is a machine learning algorithm from the Scikit-learn library. The Isolation Forest algorithm is used for anomaly detection.
- `sklearn.preprocessing.StandardScaler`: This is a utility from the Scikit-learn library that standardizes features by removing the mean and scaling to unit variance.
- `joblib`: This is a set of tools to provide lightweight pipelining in Python. Here, `dump` and `load` will be used to save the trained model to disk and load it back when needed.

Python Snippet 1: Loading the Data

```python
normal_df = pd.read_csv('data/mito-normal-dataset.csv')
```

_Explanation:_ This step involved collecting for weeks and loading the data, representing the normal operational state of my MiTo.

## Standardizing for Consistency

Python Snippet 2: Data Standardization

```python
scaler = StandardScaler()
scaled_normal_df = scaler.fit_transform(normal_df)
```

_Explanation:_ Standardization ensures that each parameter contributes equally to the analysis.

## The Learning Phase

Python Snippet 3: Model Training

```python
iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
iso_forest.fit(scaled_normal_df)
```

_Explanation:_ The model learns what normal looks like for my car’s operations.

## Identifying the Anomalies

Python Snippet 4: Loading the faulty data, a daily habit!

```python
faulty_df = pd.read_csv('data/mito-faulty-dataset.csv')
faulty_df.head()
```

![](https://miro.medium.com/v2/resize:fit:700/1*JRwLYE5_qFEmt_uv4L8qmg.png)

_Explanation:_ The features that I selected to show is the engine speed (rpm), the fuel injectors flow rate (ml/min), pulse width (milliseconds), timing (degrees), and the fuel pressure (bar).

Python Snippet 5: Anomaly Detection

```python
## Convert the 'Datetime' column to datetime
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
```

_Explanation:_ Applying the model to new data helped flag potential issues.

# Visual Insights

Python Snippet 6: Result Visualization

```python
fig, axs = plt.subplots(2, 1, figsize=(10,18))

# Plot for Anomaly Score and Injector Flow Rate
axs[0].plot(faulty_df['Datetime'], faulty_df['Anomaly_Score_IFR_Norm'], label='Anomaly Score (normalized)')
axs[0].plot(faulty_df['Datetime'], faulty_df['Injector_Flow_Rate'], label='Injector Flow Rate', linestyle='--')
axs[0].set_xlabel('Datetime')
axs[0].set_ylabel('Values')
axs[0].set_title('Anomaly Score and Injector Flow Rate over Datetime')
axs[0].legend()

# Plot for Injector Pulse Width
axs[1].plot(faulty_df['Datetime'], faulty_df['Anomaly_Score_IPW_Norm'], label='Anomaly Score (normalized)')
axs[1].plot(faulty_df['Datetime'], faulty_df['Injector_Pulse_Width'], label='Injector Pulse Width', linestyle='--')
axs[1].set_xlabel('Datetime')
axs[1].set_ylabel('Values')
axs[1].set_title('Anomaly Score and Injector Pulse Width over Datetime')
axs[1].legend()

plt.tight_layout()
plt.show()
```

![](https://miro.medium.com/v2/resize:fit:700/1*ejlZtQuG1uEypstPl7RQ9Q.png)

## Understanding the Graphs

Top Graph: Anomaly Score and Injector Flow Rate

- Anomaly Score: This is shown in blue. An anomaly score is a number that tells you how much a data point is different from a pattern. A higher score means something is more unusual. The score has been normalized to achieve better visibility.
- Injector Flow Rate: This is shown in orange with dashed lines. It tells you how fast fuel is being pushed into the engine.

Bottom Graph: Anomaly Score and Injector Pulse Width

- Injector Pulse Width: This is shown in orange with dashed lines. It measures how long the fuel injector stays open to let fuel into the engine.

## Analysis of the Data

- Starting Cold: When a car starts cold, the engine needs more fuel to run smoothly. This is because cold fuel doesn’t vaporize as well, which can make it harder for the engine to work properly.

Early Anomalies Indicated:

- In the beginning, both graphs show high anomaly scores. This suggests that the car’s fuel injection system isn’t behaving as expected.
- The Injector Flow Rate and Pulse Width may be higher than normal as the car’s systems try to adjust to the cold start.

## Implications

Potential Issues: The high anomaly scores could mean there are issues such as:

- Clogged fuel injectors that can’t provide the right amount of fuel.
- Sensors that aren’t reading temperatures correctly and cause the car to adjust fuel improperly.
- Problems with the fuel itself, like if it’s too thick because of the cold.

## Results: Fuel Injector Analysis

The comprehensive examination of the vehicle’s telemetry data, particularly focusing on the fuel injection system, has revealed critical insights pertinent to the functionality of the fuel injectors. The analysis conducted is corroborated by the official car mechanic report.

The official car mechanic report has determined that the fuel injectors are exhibiting signs of significant clogging. This assessment is based on the physical inspection of the injection system.

## Conclusion: Merging Passion with Predictive Technology

This project transcended the boundaries of a mere technical exercise; it was a harmonious blend of my enthusiasm for automobiles and my expertise in data analysis. By integrating data science into the everyday operation of my Alfa Romeo MiTo, I not only indulged in my love for cars but also took a proactive approach to vehicle maintenance.
