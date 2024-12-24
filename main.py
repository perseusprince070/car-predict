# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.multioutput import MultiOutputRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import joblib
# from fastapi import FastAPI
# from pydantic import BaseModel

# def load_and_prepare_data(filepath):
#     data = pd.read_csv(filepath)

#     data = data.dropna(subset=[
#         'price_new', 'price_dealer', 'price_trade_retail', 'price_private_clean',
#         'price_private_average', 'price_part_exchange', 'price_auction',
#         'price_trade_average', 'price_trade_poor'
#     ])

#     features = [
#         'price_new', 'price_dealer', 'price_trade_retail', 'price_private_clean',
#         'price_private_average', 'price_part_exchange', 'price_auction',
#         'price_trade_average', 'price_trade_poor'
#     ]
#     targets = [
#         'price_new', 'price_dealer', 'price_trade_retail', 'price_private_clean',
#         'price_private_average', 'price_part_exchange', 'price_auction',
#         'price_trade_average', 'price_trade_poor'
#     ]

#     X = data[features]
#     y = data[targets]

#     return X, y, targets

# def train_model(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = MultiOutputRegressor(RandomForestRegressor(random_state=42, n_estimators=100))
#     model.fit(X_train, y_train)

#     y_pred = model.predict(X_test)
#     mae = mean_absolute_error(y_test, y_pred, multioutput='uniform_average')
#     mse = mean_squared_error(y_test, y_pred, multioutput='uniform_average')
#     r2 = r2_score(y_test, y_pred, multioutput='uniform_average')

#     print("Model Performance:")
#     print(f"MAE: {mae}")
#     print(f"MSE: {mse}")
#     print(f"R2 Score: {r2}")

#     return model, list(X_train.columns)

# def save_model_and_features(model, features, model_filepath, features_filepath):
#     joblib.dump(model, model_filepath)
#     joblib.dump(features, features_filepath)

# class CarInput(BaseModel):
#     price_new: float
#     price_dealer: float
#     price_trade_retail: float
#     price_private_clean: float
#     price_private_average: float
#     price_part_exchange: float
#     price_auction: float
#     price_trade_average: float
#     price_trade_poor: float

# app = FastAPI()

# @app.post("/predict")
# def predict_car_values(car: CarInput):
#     try:
#         model = joblib.load("car_valuation_model.pkl")
#         feature_names = joblib.load("model_features.pkl")
#     except FileNotFoundError:
#         return {"error": "Model or feature file not found. Train the model first."}

#     input_data = pd.DataFrame([{
#         'price_new': car.price_new,
#         'price_dealer': car.price_dealer,
#         'price_trade_retail': car.price_trade_retail,
#         'price_private_clean': car.price_private_clean,
#         'price_private_average': car.price_private_average,
#         'price_part_exchange': car.price_part_exchange,
#         'price_auction': car.price_auction,
#         'price_trade_average': car.price_trade_average,
#         'price_trade_poor': car.price_trade_poor
#     }])

#     input_data = input_data.reindex(columns=feature_names, fill_value=0)
#     predicted_values = model.predict(input_data)[0]
#     output = {col: predicted_values[i] for i, col in enumerate(feature_names)}

#     return output

# if __name__ == "__main__":
#     dataset_path = "datavalue_2024-12-17.csv"
#     try:
#         X, y, targets = load_and_prepare_data(dataset_path)
#         trained_model, feature_names = train_model(X, y)

#         save_model_and_features(trained_model, feature_names, "car_valuation_model.pkl", "model_features.pkl")

#         print("Model training and saving completed.")
#     except FileNotFoundError:
#         print("Dataset not found. Ensure the dataset is available at the specified path.")


import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load and prepare data
def load_and_prepare_data(filepath):
    data = pd.read_csv(filepath)

    # Drop rows with missing target values
    data = data.dropna(subset=[
        'price_new', 'price_dealer', 'price_trade_retail', 'price_private_clean',
        'price_private_average', 'price_part_exchange', 'price_auction',
        'price_trade_average', 'price_trade_poor'
    ])

    # Define features and targets
    features = ['year', 'age', 'miles']
    targets = [
        'price_new', 'price_dealer', 'price_trade_retail', 'price_private_clean',
        'price_private_average', 'price_part_exchange', 'price_auction',
        'price_trade_average', 'price_trade_poor'
    ]

    X = data[features]
    y = data[targets]

    return X, y, targets

# Train model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultiOutputRegressor(RandomForestRegressor(random_state=42, n_estimators=100))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred, multioutput='uniform_average')
    mse = mean_squared_error(y_test, y_pred, multioutput='uniform_average')
    r2 = r2_score(y_test, y_pred, multioutput='uniform_average')

    print("Model Performance:")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"R2 Score: {r2}")

    return model, list(X_train.columns)

# Save model and features
def save_model_and_features(model, features, model_filepath, features_filepath):
    joblib.dump(model, model_filepath)
    joblib.dump(features, features_filepath)

# Command-line interface
if __name__ == "__main__":
    dataset_path = "datavalue.csv"

    # If arguments are provided, make predictions
    if len(sys.argv) > 4:
        try:
            # Load trained model and feature names
            model = joblib.load("car_valuation_model.pkl")
            feature_names = joblib.load("model_features.pkl")
        except FileNotFoundError:
            print("Error: Model or feature file not found. Train the model first.")
            sys.exit(1)

        # Parse command-line arguments
        make = sys.argv[1]  # Example: make is not used in this version
        model_name = sys.argv[2]  # Example: model_name is not used in this version
        year = int(sys.argv[3])
        mileage = float(sys.argv[4])

        # Create input data
        input_data = pd.DataFrame([{
            'year': year,
            'age': 2024 - year,  # Assuming current year is 2024
            'miles': mileage
        }])

        # Reindex to match trained model's feature names
        input_data = input_data.reindex(columns=feature_names, fill_value=0)

        # Predict
        predicted_values = model.predict(input_data)[0]
        print("Predicted Values:")
        for i, target in enumerate([
            'price_new', 'price_dealer', 'price_trade_retail', 'price_private_clean',
            'price_private_average', 'price_part_exchange', 'price_auction',
            'price_trade_average', 'price_trade_poor'
        ]):
            print(f"{target}: {predicted_values[i]:.2f}")

    else:
        # Train the model if no arguments are provided
        try:
            X, y, targets = load_and_prepare_data(dataset_path)
            trained_model, feature_names = train_model(X, y)

            save_model_and_features(trained_model, feature_names, "car_valuation_model.pkl", "model_features.pkl")

            print("Model training and saving completed.")
        except FileNotFoundError:
            print("Error: Dataset not found. Ensure the dataset is available at the specified path.")
