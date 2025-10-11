from config import *

def main():
    # Load data from SQLite database
    df = pd.read_sql_query("SELECT * FROM usedCars", source_conn)

    print(f"Vehicles: {len(df)}, Missing prices: {df['price'].isnull().sum()} "
          f"({df['price'].isnull().sum() / len(df) * 100:.1f}%)")

    # ULTIMATE VERSION - OPTIMIZED
    def preprocess_data_ultimate(df):
        df_cleaned = df.copy()

        # 1. STRICT CLEANING
        df_cleaned = df_cleaned.dropna(subset=['price'])

        # 2. OUTLIER REMOVAL
        q05 = df_cleaned['price'].quantile(0.05)
        q95 = df_cleaned['price'].quantile(0.95)
        df_cleaned = df_cleaned[(df_cleaned['price'] >= q05) & (df_cleaned['price'] <= q95)]

        print(f"After cleaning: {len(df_cleaned)} vehicles")

        if len(df_cleaned) < 10:
            print(f"Not enough data after cleaning ({len(df_cleaned)} vehicles).")
            return

        # 3. ADVANCED FEATURE ENGINEERING
        current_year = 2025
        df_cleaned['car_age'] = current_year - df_cleaned['year']
        df_cleaned['mileage_per_year'] = df_cleaned['mileage'] / np.where(df_cleaned['car_age'] == 0, 1,
                                                                          df_cleaned['car_age'])

        # Binary strategic indicators
        premium_brands = [
            'MERCEDES-BENZ', 'BMW', 'AUDI', 'PORSCHE', 'LAND-ROVER',
            'JAGUAR', 'VOLVO', 'LEXUS', 'BENTLEY', 'MASERATI',
            'FERRARI', 'ASTON MARTIN', 'TESLA', 'CADILLAC'
        ]

        df_cleaned['is_premium'] = df_cleaned['brand'].isin(premium_brands).astype(int)
        df_cleaned['is_diesel'] = (df_cleaned['fuel_type'] == 'Diesel').astype(int)
        df_cleaned['is_automatic'] = (df_cleaned['transmission'] == 'Automatic').astype(int)
        df_cleaned['is_hybrid'] = (df_cleaned['fuel_type'] == 'Hybrid').astype(int)
        df_cleaned['is_suv'] = df_cleaned['model'].str.contains(
            'Q[0-9]|X[0-9]|GLC|X3|X5|Tucson|Sportage|Kuga|3008|5008',
            na=False
        ).astype(int)

        # 4. EFFICIENT ENCODING
        categorical_cols = ['brand', 'model', 'transmission', 'fuel_type']
        label_encoders = {}

        for col in categorical_cols:
            le = LabelEncoder()
            df_cleaned[col] = le.fit_transform(df_cleaned[col].astype(str))
            label_encoders[col] = le

        # Final features
        features = [
            'brand', 'model', 'year', 'mileage', 'transmission', 'fuel_type',
            'car_age', 'mileage_per_year', 'is_premium', 'is_diesel',
            'is_automatic', 'is_hybrid', 'is_suv'
        ]

        X = df_cleaned[features]
        y = df_cleaned['price']

        return X, y, label_encoders, df_cleaned

    # DATA PREPARATION
    X, y, encoders, df_cleaned = preprocess_data_ultimate(df)

    print("\nFINAL STATISTICS:")
    print(f"Vehicles: {len(X)}")
    print(f"Price - Min: {y.min():,.0f} MAD, Max: {y.max():,.0f} MAD")
    print(f"Price - Median: {y.median():,.0f} MAD, Mean: {y.mean():,.0f} MAD")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # Normalization
    scaler = StandardScaler()
    num_cols = ['year', 'mileage', 'car_age', 'mileage_per_year']
    X_train.loc[:, num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test.loc[:, num_cols] = scaler.transform(X_test[num_cols])

    # CATBOOST TRAINING
    print("\nTraining the model...")
    final_model = CatBoostRegressor(
        iterations=500,
        depth=8,
        learning_rate=0.05,
        l2_leaf_reg=3,
        random_state=42,
        verbose=100,
        early_stopping_rounds=50
    )

    final_model.fit(X_train, y_train, eval_set=(X_test, y_test), plot=False)

    # MODEL EVALUATION
    y_pred = final_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    def safe_mape(y_true, y_pred):
        mask = (y_true != 0) & (y_true > 1000)
        if mask.sum() > 0:
            return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return float('inf')

    mape = safe_mape(y_test, y_pred)
    mad_to_eur = 0.091

    print("\nMODEL PERFORMANCE:")
    print(f"R² Score: {r2:.3f}")
    print(f"MAE: {mae:,.0f} MAD ({mae * mad_to_eur:,.0f} €)")
    print(f"RMSE: {rmse:,.0f} MAD ({rmse * mad_to_eur:,.0f} €)")
    print(f"MAPE: {mape:.1f}%")
    print(f"Accuracy: {max(0, 100 - mape):.1f}%")

    # FEATURE IMPORTANCE
    importance = final_model.feature_importances_
    importance_pct = 100 * importance / importance.sum()

    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance (%)': importance_pct
    }).sort_values('Importance (%)', ascending=False)

    print("\nTOP 10 MOST IMPORTANT FEATURES:")
    print(feature_importance.head(10).to_string(index=False))

    # PREDICTION FUNCTION
    def predict_car_price_ultimate(brand, model, year, mileage, transmission, fuel_type):
        try:
            car_data = {
                'brand': brand, 'model': model, 'year': year,
                'mileage': mileage, 'transmission': transmission, 'fuel_type': fuel_type
            }

            df_car = pd.DataFrame([car_data])
            current_year = 2025
            df_car['car_age'] = current_year - df_car['year']
            df_car['mileage_per_year'] = df_car['mileage'] / np.where(df_car['car_age'] == 0, 1, df_car['car_age'])

            premium_brands = [
                'MERCEDES-BENZ', 'BMW', 'AUDI', 'PORSCHE', 'LAND-ROVER',
                'JAGUAR', 'VOLVO', 'LEXUS', 'BENTLEY', 'MASERATI',
                'FERRARI', 'ASTON MARTIN', 'TESLA', 'CADILLAC'
            ]

            df_car['is_premium'] = 1 if brand in premium_brands else 0
            df_car['is_diesel'] = 1 if fuel_type == 'Diesel' else 0
            df_car['is_automatic'] = 1 if transmission == 'Automatic' else 0
            df_car['is_hybrid'] = 1 if fuel_type == 'Hybrid' else 0
            df_car['is_suv'] = 1 if any(term in model.upper() for term in
                                        ['Q3', 'Q5', 'Q7', 'Q8', 'X3', 'X5', 'X6', 'X7', 'GLC', 'TUCSON', 'SPORTAGE',
                                         '3008', '5008', 'KUGA']) else 0

            for col in ['brand', 'model', 'transmission', 'fuel_type']:
                if col in encoders:
                    if car_data[col] in encoders[col].classes_:
                        df_car[col] = encoders[col].transform([car_data[col]])[0]
                    else:
                        df_car[col] = len(encoders[col].classes_) // 2
                else:
                    df_car[col] = 0

            df_car.loc[:, num_cols] = scaler.transform(df_car[num_cols])

            price_mad = final_model.predict(df_car)[0]
            price_eur = price_mad * mad_to_eur

            return max(1000, price_mad), price_eur

        except Exception as e:
            print(f"Prediction error: {e}")
            return None, None

    # TEST EXAMPLES
    print("\nREFERENCE PREDICTIONS:")

    test_cars = [
        {'brand': 'RENAULT', 'model': 'Clio', 'year': 2020, 'mileage': 40000, 'transmission': 'Manual',
         'fuel_type': 'Diesel'},
        {'brand': 'VOLKSWAGEN', 'model': 'Golf 7', 'year': 2018, 'mileage': 60000, 'transmission': 'Manual',
         'fuel_type': 'Diesel'},
        {'brand': 'DACIA', 'model': 'Sandero', 'year': 2022, 'mileage': 20000, 'transmission': 'Manual',
         'fuel_type': 'Gasoline'},
        {'brand': 'MERCEDES-BENZ', 'model': 'Classe C', 'year': 2021, 'mileage': 25000, 'transmission': 'Automatic',
         'fuel_type': 'Diesel'},
        {'brand': 'PEUGEOT', 'model': '3008', 'year': 2020, 'mileage': 45000, 'transmission': 'Automatic',
         'fuel_type': 'Diesel'}
    ]

    for car in test_cars:
        price_mad, price_eur = predict_car_price_ultimate(**car)
        if price_mad:
            print(f" {car['brand']:12} {car['model']:10} {car['year']} | {car['mileage']:6,} km")
            print(f"  {price_mad:>8,.0f} MAD  ≈ {price_eur:>7,.0f} EUR\n")

    # FINAL REPORT
    print("\nFINAL MODEL REPORT:")
    print(f"   - {len(X):,} vehicles analyzed")
    print(f"   - R² = {r2:.3f}")
    print(f"   - Mean Absolute Error: {mae:,.0f} MAD")
    print(f"   - Accuracy: {max(0, 100 - mape):.1f}%")

if __name__ == "__main__":
    main()