from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__, template_folder='Flask/templates' ,static_folder='Flask/static')

# Load your pre-trained model
with open('./Saved_Models/lightgbm.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Open the scaler file and load it
with open('./Saved_Models/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    type_input = request.form['type']
    air_temp = float(request.form['air_temp'])
    process_temp = float(request.form['process_temp'])
    rotational_speed = float(request.form['rotational_speed'])
    torque = float(request.form['torque'])
    tool_wear = float(request.form['tool_wear'])

    # Feature extraction
    power = rotational_speed * torque
    temp_diff = process_temp - air_temp
    tool_wear_torque_product = tool_wear * torque

    # Create DataFrame with the necessary columns for prediction
    df = pd.DataFrame({
        'Type': [type_input],
        'Air temperature [K]': [air_temp],
        'Process temperature [K]': [process_temp],
        'Rotational speed [rpm]': [rotational_speed],
        'Torque [Nm]': [torque],
        'Tool wear [min]': [tool_wear],
        'Power': [power],
        'Temp_diff': [temp_diff],
        'tool_wear_torque_product': [tool_wear_torque_product]
    })


    # Preprocessing and scaling
    # Select the same columns used for training
    type_mapping = {'L': 0, 'M': 1, 'H': 2}
    df['Type'] = df['Type'].map(type_mapping)

    features_for_scaling = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 
                'Torque [Nm]', 'Tool wear [min]', 
                'Power', 'Temp_diff', 'tool_wear_torque_product']


    df_scaled = scaler.transform(df[features_for_scaling])
    df_scaled_features = pd.DataFrame(df_scaled, columns=features_for_scaling).reset_index(drop=True)

    # Remove anything inside brackets []
    df_scaled_features.columns = df_scaled_features.columns.str.replace(r'\[.*?\]', '', regex=True)  


    #df_scaled = scaler.transform(df[columns_for_model])
    df_combined = df[['Type']].reset_index(drop=True)
    # Recombine 'Type' and scaled features
    final_df = pd.concat([df_combined, df_scaled_features], axis=1)

    # Make prediction
    prediction = model.predict(final_df)
    result = int(prediction[0])  # 0 for No Failure, 1 for Failure

    return jsonify({'prediction': result })

if __name__ == '__main__':
    app.run(debug=True)
