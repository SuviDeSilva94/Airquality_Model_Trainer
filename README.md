# Air Quality Prediction API

This project provides an API to predict air quality (AQI) for the next 24 hours using historical air quality data and machine learning. The API uses data from OpenWeatherMap and a Random Forest model to generate predictions for cities based on PM2.5 and PM10 values.

## Technologies Used

- **Python**: The main programming language used for developing the application.
- **Flask**: A lightweight Python web framework used to build the RESTful API.
- **OpenWeatherMap API**: Used for retrieving real-time and historical air quality data.
- **Machine Learning (scikit-learn)**: A machine learning library used to train a Random Forest model on historical air quality data.
- **joblib**: Used to save and load the trained machine learning model.
- **Pandas**: Used to handle and process data in tabular format.
- **NumPy**: Used for numerical operations and data handling.
- **Flask-CORS**: A Flask extension to enable Cross-Origin Resource Sharing (CORS) for API access.
- **Datetime**: Used for handling date and time manipulation for historical data retrieval.

## Features

- Fetches the latitude and longitude for a city using the OpenWeatherMap Geocoding API.
- Retrieves historical air quality data for the past week from OpenWeatherMap.
- Trains a Random Forest model on historical air quality data.
- Predicts AQI for the next 24 hours using the trained model and calculates the AQI from PM2.5 and PM10 values.
- Returns AQI prediction data along with AQI status (Good, Moderate, Unhealthy, etc.).
- CORS-enabled for seamless API integration with frontend applications.

## Requirements

- **Python 3.7+**
- **Flask** (Flask web framework)
- **Flask-CORS** (For enabling CORS)
- **requests** (For making HTTP requests to external APIs)
- **scikit-learn** (For machine learning model training and predictions)
- **joblib** (For saving/loading the machine learning model)
- **pandas** (For data manipulation)
- **numpy** (For numerical operations)
- **datetime** (For handling time data)

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/air-quality-prediction-api.git
   cd air-quality-prediction-api
   ```

2. **Set up a virtual environment (optional but recommended)**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Get OpenWeatherMap API Key**:

   - Go to [OpenWeatherMap](https://openweathermap.org/) and create an account to get an API key.
   - Replace `your_api_key` in the code with your actual API key.

5. **Train the Machine Learning Model** (Optional):

   - The model is trained using historical data for the past 7 days. You can modify this behavior based on the data available. The model training is done automatically in the `predict_aqi` endpoint.

6. **Run the Flask Application**:

   ```bash
   python app.py
   ```

   The Flask server will start running locally on port `5001` (by default). You can access the API at `http://127.0.0.1:5001`.

## API Usage

The API exposes a POST endpoint `/predict` to get AQI predictions for a city.

### Request Body

Send a POST request to `/predict` with the following JSON body:

```json
{
  "city": "London"
}
```

- **city** (required): Name of the city for which you want to predict AQI.

### Response

The response will contain the AQI prediction for the next 24 hours, including the PM2.5 and PM10 values, AQI value, and AQI status for each hour.

Example response:

```json
{
  "City": "London",
  "Predictions": [
    {
      "hour": 0,
      "PM2.5": 12.5,
      "PM10": 45.0,
      "AQI": 48.0,
      "AQI Status": "Good"
    },
    {
      "hour": 1,
      "PM2.5": 13.0,
      "PM10": 50.0,
      "AQI": 52.3,
      "AQI Status": "Good"
    },
    ...
  ]
}
```

### Error Response

If there is an error (e.g., city not found, API failure), you will get an error message in the response:

```json
{
  "error": "Unable to find coordinates for city: London"
}
```

## Example Use Cases

- **Frontend Integration**: This API can be used in a React, Angular, or Vue.js application to display real-time air quality forecasts for cities.
- **Mobile Apps**: You can integrate this API into a mobile app (React Native, Flutter) to show air quality predictions to users.
- **Environmental Studies**: The model can be used for academic or research purposes to analyze and predict air quality levels.

## How the Model Works

1. The historical air quality data for a given city is fetched using the OpenWeatherMap API.
2. The data is preprocessed, and a Random Forest Regressor model is trained on the PM2.5 and PM10 values.
3. The trained model predicts AQI for the next 24 hours.
4. The AQI is calculated for both PM2.5 and PM10 values using respective formulas.
5. The highest AQI is selected as the final prediction for each hour, and the AQI status is derived based on the value.

## Contributing

If you'd like to contribute to the project, feel free to fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

- Your Name – [Suvi De Silva](https://github.com/SuviDeSilva94)

