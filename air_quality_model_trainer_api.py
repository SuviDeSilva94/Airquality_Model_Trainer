from flask import Flask, request, jsonify
import requests
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import datetime
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# OpenWeatherMap API Key and URLs (replace 'your_api_key' with your actual key)
API_KEY = ''
GEOCODING_API_URL = 'http://api.openweathermap.org/data/2.5/weather'
AIR_QUALITY_API_URL = 'http://api.openweathermap.org/data/2.5/air_pollution'
HISTORICAL_API_URL = 'http://api.openweathermap.org/data/2.5/air_pollution/history'

# Step 1: Get latitude and longitude from city name using OpenWeatherMap Geocoding API
def get_coordinates_from_city(city_name):
    params = {
        'q': city_name,
        'appid': API_KEY
    }
    response = requests.get(GEOCODING_API_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        lat = data['coord']['lat']
        lon = data['coord']['lon']
        return lat, lon
    else:
        return None, None

# Step 2: Fetch historical air quality data for the past N hours
def get_historical_air_quality_data(lat, lon, start_time, end_time):
    params = {
        'lat': lat,
        'lon': lon,
        'start': start_time,
        'end': end_time,
        'appid': API_KEY
    }
    response = requests.get(HISTORICAL_API_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        air_quality_data = []
        for entry in data['list']:
            pm25 = entry['components'].get('pm2_5', None)
            pm10 = entry['components'].get('pm10', None)
            if pm25 is not None and pm10 is not None:
                air_quality_data.append({
                    'PM2.5': pm25,
                    'PM10': pm10,
                    'time': datetime.datetime.utcfromtimestamp(entry['dt']).strftime('%Y-%m-%d %H:%M:%S')
                })
        return air_quality_data
    else:
        return None

# Correct AQI Calculation for PM2.5
def calculate_aqi_pm25(pm25):
    if pm25 <= 12:
        aqi = (50 / 12) * pm25
    elif pm25 <= 35.4:
        aqi = ((100 - 51) / (35.4 - 12.1)) * (pm25 - 12.1) + 51
    elif pm25 <= 55.4:
        aqi = ((150 - 101) / (55.4 - 35.5)) * (pm25 - 35.5) + 101
    elif pm25 <= 150.4:
        aqi = ((200 - 151) / (150.4 - 55.5)) * (pm25 - 55.5) + 151
    elif pm25 <= 250.4:
        aqi = ((300 - 201) / (250.4 - 150.5)) * (pm25 - 150.5) + 201
    elif pm25 <= 350.4:
        aqi = ((400 - 301) / (350.4 - 250.5)) * (pm25 - 250.5) + 301
    else:
        aqi = ((500 - 401) / (500.4 - 350.5)) * (pm25 - 350.5) + 401
    return aqi

# Correct AQI Calculation for PM10
def calculate_aqi_pm10(pm10):
    if pm10 <= 54:
        aqi = (50 / 54) * pm10
    elif pm10 <= 154:
        aqi = ((100 - 51) / (154 - 55)) * (pm10 - 55) + 51
    elif pm10 <= 254:
        aqi = ((150 - 101) / (254 - 155)) * (pm10 - 155) + 101
    elif pm10 <= 354:
        aqi = ((200 - 151) / (354 - 255)) * (pm10 - 255) + 151
    elif pm10 <= 424:
        aqi = ((300 - 201) / (424 - 355)) * (pm10 - 355) + 201
    elif pm10 <= 504:
        aqi = ((400 - 301) / (504 - 425)) * (pm10 - 425) + 301
    else:
        aqi = ((500 - 401) / (604 - 505)) * (pm10 - 505) + 401
    return aqi

# Step 3: Train model (this is for the sake of completeness, it is optional)
def train_model(historical_data):
    df = pd.DataFrame(historical_data)
    df['time'] = pd.to_datetime(df['time'])  # Convert time to datetime
    df = df.set_index('time')  # Set time as the index

    # Prepare features and target variables
    X = df[['PM2.5', 'PM10']]  # Features (PM2.5 and PM10)
    y = np.random.rand(len(df)) * 300  # Dummy AQI values for now, replace with real AQI data

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Save the trained model to a file
    joblib.dump(model, 'aqi_model.pkl')
    print(f'Model saved to aqi_model.pkl')

# Step 4: Load model
def load_model():
    return joblib.load('aqi_model.pkl')

# Step 5: AQI status function
def get_aqi_status(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

# Step 6: API endpoint to get AQI prediction for the next 24 hours
@app.route('/predict', methods=['POST'])
def predict_aqi():
    data = request.get_json()
    city_name = data.get('city')

    if city_name is None:
        return jsonify({'error': 'City name is required'}), 400

    # Get latitude and longitude of the city using Geocoding API
    lat, lon = get_coordinates_from_city(city_name)
    if lat is None or lon is None:
        return jsonify({'error': f'Unable to find coordinates for city: {city_name}'}), 500

    # Get the current time
    current_time = datetime.datetime.utcnow()

    # Define the time range for historical data (e.g., past 7 days)
    start_time = int((current_time - datetime.timedelta(days=7)).timestamp())
    end_time = int(current_time.timestamp())

    # Fetch historical air quality data from OpenWeatherMap
    historical_data = get_historical_air_quality_data(lat, lon, start_time, end_time)
    if not historical_data:
        return jsonify({'error': 'Unable to fetch historical air quality data'}), 500

    # Train the model using the historical data
    train_model(historical_data)

    # Load the trained model
    model = load_model()

    # Predict AQI for the next 24 hours
    prediction_data = []
    for hour in range(24):
        # Use the PM2.5 and PM10 values from historical data or random for testing
        pm25 = historical_data[hour % len(historical_data)]['PM2.5']
        pm10 = historical_data[hour % len(historical_data)]['PM10']
        
        # Predict AQI using trained model
        aqi = model.predict([[pm25, pm10]])[0]
        
        # Calculate the AQI using the correct formula
        aqi_pm25 = calculate_aqi_pm25(pm25)
        aqi_pm10 = calculate_aqi_pm10(pm10)
        aqi = max(aqi_pm25, aqi_pm10)  # Use the highest AQI from PM2.5 or PM10
        
        # Get AQI status
        aqi_status = get_aqi_status(aqi)
        
        prediction_data.append({
            'hour': hour,
            'PM2.5': pm25,
            'PM10': pm10,
            'AQI': round(aqi, 2),
            'AQI Status': aqi_status
        })

    return jsonify({
        'City': city_name,
        'Predictions': prediction_data
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
