import joblib
import pandas as pd

def load_model(model_path="model/passenger_load_model.joblib"):
    """Load the trained ML model from file."""
    return joblib.load(model_path)

def predict(model, input_data):
    """Run prediction on new input data."""
    prediction = model.predict(input_data)
    return prediction

def main():
    # Load model
    model = load_model()

    # Example input data
    example_data = pd.DataFrame([{
        "departures_performed": 10,
        "payload": 5000,
        "freight": 300,
        "mail": 50,
        "distance": 1200,
        "air_time": 150,
        "carrier_group": "Major",
        "aircraft_type": "Boeing 737-800",
        "aircraft_config": "2-class",
        "route_type": "Domestic"
    }])

    # Predict passenger load factor
    prediction = predict(model, example_data)
    print(f"Predicted Load Factor: {prediction[0]:.3f}")

if __name__ == "__main__":
    main()

