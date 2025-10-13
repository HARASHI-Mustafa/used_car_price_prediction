from config import *
from price_prediction import predict_car_price_ultimate
from catboost import CatBoostRegressor
from joblib import load
import pandas as pd

def main():
    # Load the model
    print("Loading model...")
    model = CatBoostRegressor()
    model.load_model("catboost_model.cbm")

    # Load encoders and scaler
    encoders = load("encoders.joblib")
    scaler = load("scaler.joblib")

    # Test examples
    print("REFERENCE PREDICTIONS\n")

    test_cars = [
        {'brand': 'RENAULT', 'model': 'Clio', 'year': 2020, 'mileage': 40000, 'transmission': 'Manual', 'fuel_type': 'Diesel'},
        {'brand': 'VOLKSWAGEN', 'model': 'Golf 7', 'year': 2018, 'mileage': 60000, 'transmission': 'Manual', 'fuel_type': 'Diesel'},
        {'brand': 'DACIA', 'model': 'Sandero', 'year': 2022, 'mileage': 20000, 'transmission': 'Manual', 'fuel_type': 'Gasoline'},
        {'brand': 'MERCEDES-BENZ', 'model': 'Classe C', 'year': 2021, 'mileage': 25000, 'transmission': 'Automatic', 'fuel_type': 'Diesel'},
        {'brand': 'PEUGEOT', 'model': '3008', 'year': 2020, 'mileage': 45000, 'transmission': 'Automatic', 'fuel_type': 'Diesel'}
    ]

    def predict_single_car(car_dict):
        df_car = pd.DataFrame([car_dict])
        predicted = predict_car_price_ultimate(df_car, model, encoders, scaler)[0]
        mad_to_eur = 0.091
        return predicted, predicted * mad_to_eur

    for car in test_cars:
        price_mad, price_eur = predict_single_car(car)
        print(f"{car['brand']:12} {car['model']:10} {car['year']} | {car['mileage']:6,} km")
        print(f"  {price_mad:>8,.0f} MAD  ≈ {price_eur:>7,.0f} EUR\n")


if __name__ == "__main__":
    main()