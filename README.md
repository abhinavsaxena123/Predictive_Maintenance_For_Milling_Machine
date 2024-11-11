# Predictive_Maintenance_For_Milling_Machine ⚙️

## Overview 🪄
Predictive Maintenance for Milling Machines is a project aimed at utilizing data-driven techniques to anticipate equipment failures before they occur. By applying machine learning models to sensor data, we can reduce machine downtime, optimize maintenance schedules, and extend machine lifespan.

## Background
Milling machines are key assets in manufacturing but are prone to wear and tear, leading to unexpected breakdowns and expensive repairs. Predictive maintenance enables us to leverage historical and real-time data to predict when maintenance is needed, minimizing downtime and avoiding catastrophic failures.

## Objective
The main goal of this project is to develop a predictive model that can accurately forecast the failure of milling machines based on sensor data, environmental factors, and historical maintenance records. 
This model will help:
* Reduce unexpected machine downtime.
* Optimize maintenance schedules.
* Minimize operational costs.

### Project Structure 📒
```
Predictive_Maintenance_For_Milling_Machine/
├── Data/
│   ├── ai4i2020.csv             # Raw data files
│   ├── processed/               # Cleaned and processed data files
├── Flask/
│   └── static                   #images
|   └── templates                #index.html
│── Saved_Models/                # .pkl files
│── EDA.ipnb                       
|── Model_Building.ipynb
├── app.py                       #Flask app
├── README.md                    # Project overview
├── requirements.txt             # Project dependencies
```

## Data Description: 
Dataset source: Kaggle, UCI
* Dataset Link: https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020

This synthetic dataset is modeled after an existing milling machine and consists of 10 000 data points from a stored as rows with 14 features in columns.
1. UID: unique identifier ranging from 1 to 10000
2. product ID: consisting of a letter L, M, or H for low (50% of all products), medium (30%) and high (20%) as product quality variants and a variant-specific serial number
3. type: just the product type L, M or H from column 2
4. air temperature [K]: generated using a random walk process later normalized to a standard deviation of 2 K around 300 K
5. process temperature [K]: generated using a random walk process normalized to a standard deviation of 1 K, added to the air temperature plus 10 K.
6. rotational speed [rpm]: calculated from a power of 2860 W, overlaid with a normally distributed noise
7. torque [Nm]: torque values are normally distributed around 40 Nm with a SD = 10 Nm and no negative values.
8. tool wear [min]: The quality variants H/M/L add 5/3/2 minutes of tool wear to the used tool in the process.
9. 'machine failure' label that indicates, whether the machine has failed in this particular datapoint for any of the following failure modes are true.

This dataset is part of the following publication, please cite when using this dataset:
S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications," 2020 Third International Conference on Artificial Intelligence for Industries (AI4I), 2020, pp. 69-74, doi: 10.1109/AI4I49448.2020.00023.

____________________________________________________________________________________________________________________________________________________________________________

## Installation ⚙️
### 1. Clone the repository:
```
git clone https://github.com/abhinavsaxena123/Predictive_Maintenance_For_Milling_Machine
cd Predictive_Maintenance_For_Milling_Machine
```

### 2. Set Up a Virtual Environment
* Using venv (for Python):
```
python3 -m venv env                        # Use `python` on Windows instead of `python3`
source env/bin/activate                    # On Windows, use source env/Scripts/activate
```

### 3. Install dependencies:
```
pip install -r requirements.txt
```

## Usage
### 1. Run Your Flask Application:
```
python app.py
```
* Access the application: Open your web browser and go to http://127.0.0.1:5000 to access the main page of the app.
* Deactivate the Virtual Environment: After you’re done working, deactivate the virtual environment by running:
deactivate


## Methodology

### 1. Data Collection and Cleaning: Acquiring machine data, handling missing values, and outliers.

### 2. Exploratory Data Analysis (EDA): Identifying patterns, correlations, and trends.

### Some Distribution Plots:
<img width="428" alt="image" src="https://github.com/user-attachments/assets/7b6a8faf-9b5b-43e9-b299-456040f9f71b">

<img width="430" alt="image" src="https://github.com/user-attachments/assets/c49c3fba-542d-4703-87d9-1e4e714a89e3">

#### Problem: Data is highly Imbalanced.
#### Our Approach: Using Models and techniques which are well-suited for handling imbalanced datasets.

<img width="900" height="300" alt="image" src="https://github.com/user-attachments/assets/e9a394e5-020e-4480-8ca9-bfff6c1232c6">
<img width="238" alt="image" src="https://github.com/user-attachments/assets/82e84a6d-97f1-4775-a5c8-b320234aacb1">

<img width="800" alt="image" src="https://github.com/user-attachments/assets/d93122fc-7e7d-40a3-b654-61e1b10dcb5a">

### Checking Distribution of Data (Q-Q Plots):
<img width="849" alt="image" src="https://github.com/user-attachments/assets/2444e2fd-675d-46a1-9461-957f7ad74adb">
<img width="864" alt="image" src="https://github.com/user-attachments/assets/f1c9f9f4-7c06-419b-8294-61cec1a24e59">
<img width="874" alt="image" src="https://github.com/user-attachments/assets/4535ea9b-c3ae-4922-a265-ca28a36a83b2">
<img width="863" alt="image" src="https://github.com/user-attachments/assets/7709acd8-6a71-42c9-8299-32c25bcaddd2">

### 3. Feature Engineering: Creating new variables from raw data, like Power, Temp_diff, tool_wear_torque_product.
```
df['Power'] = df[['Rotational speed [rpm]', 'Torque [Nm]']].product(axis=1)
```

```
df['Temp_diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
```

```
df['tool_wear_torque_product'] = df[['Tool wear [min]', 'Torque [Nm]']].product(axis=1)
```

<img width="794" alt="image" src="https://github.com/user-attachments/assets/ea50fea8-04b3-4371-a2ab-b380e1f12731">

### 4. Model Selection: Testing algorithms like Random Forest, XGBoost, and SVM etc.

### 5. Model Evaluation: Assessing model performance using metrics such as accuracy, precision, recall, F1 score, and AUC-ROC.

## Results:
<img width="841" alt="image" src="https://github.com/user-attachments/assets/e0db099f-7d61-47eb-a5e0-2f9eb2d38bc0">

### Analysis:
* Random Forest has the highest overall performance with 99.03% accuracy and strong F1, precision, and recall scores. This model also has a high PR AUC of 85.79%, indicating good performance on the precision-recall curve, especially for imbalanced classes.
* LightGBM and XGBoost are very close in performance, with LightGBM slightly outperforming XGBoost in F1 score and recall. Both models are still significantly better than the other models, though they lag behind Random Forest in terms of accuracy.
* SVM performs worse than the tree-based models, with a significantly lower precision (65.50%) and F1 score (71.47%). While recall is high (92.96%), the precision is much lower, which could indicate that the model is over-predicting the positive class. The PR AUC of 62.31% also suggests that the model's precision-recall trade-off isn't ideal.
* Logistic Regression has the lowest performance overall, with poor F1 score, precision, and PR AUC. While its recall is relatively decent at 84.46%, the model is not doing as well on the other metrics, indicating it may struggle to properly classify positive instances.

## Conclusion:
* Random Forest is the top performer in terms of overall accuracy and balanced metrics.
* LightGBM and XGBoost are also strong choices, especially if one is aiming for slightly faster models with comparable performance.
* SVM and Logistic Regression are not ideal for this specific task, with Logistic Regression performing the worst across almost all metrics.

## Future Work
Future improvements include:

* Integrating real-time data streams.
* Expanding to predictive maintenance for other machinery.
* Adding a REST API for broader integration possibilities.


## Contributors
Abhinav Saxena

