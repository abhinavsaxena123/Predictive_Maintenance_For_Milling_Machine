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
```
deactivate
```














## Future Work
Future improvements include:

* Integrating real-time data streams.
* Expanding to predictive maintenance for other machinery.
* Adding a REST API for broader integration possibilities.


## Contributors
Abhinav Saxena

