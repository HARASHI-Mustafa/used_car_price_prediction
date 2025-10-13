from config import *

# PREDICTION FUNCTION
def predict_car_price_ultimate(batch_df, model, encoders, scaler):
    batch_fe, features, _, _ = feature_engineering(batch_df, encoders=encoders, scaler=scaler, fit=False)
    predicted = model.predict(batch_fe[features])
    return predicted

# Common feature engineering for training and prediction
def feature_engineering(df_input, encoders=None, scaler=None, fit=False):
    df_out = df_input.copy()
    current_year = 2025

    # New features
    df_out['car_age'] = current_year - df_out['year']
    df_out['mileage_per_year'] = df_out['mileage'] / np.where(df_out['car_age'] == 0, 1, df_out['car_age'])

    premium_brands = [
        'MERCEDES-BENZ', 'BMW', 'AUDI', 'PORSCHE', 'LAND-ROVER',
        'JAGUAR', 'VOLVO', 'LEXUS', 'BENTLEY', 'MASERATI',
        'FERRARI', 'ASTON MARTIN', 'TESLA', 'CADILLAC'
    ]
    df_out['is_premium'] = df_out['brand'].isin(premium_brands).astype(int)
    df_out['is_diesel'] = (df_out['fuel_type'] == 'Diesel').astype(int)
    df_out['is_automatic'] = (df_out['transmission'] == 'Automatic').astype(int)
    df_out['is_hybrid'] = (df_out['fuel_type'] == 'Hybrid').astype(int)
    df_out['is_suv'] = df_out['model'].fillna('').str.upper().str.contains(
        'Q[0-9]|X[0-9]|GLC|X3|X5|TUCSON|SPORTAGE|KUGA|3008|5008'
    ).astype(int)

    categorical_cols = ['brand', 'model', 'transmission', 'fuel_type']
    if fit:
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df_out[col] = le.fit_transform(df_out[col].astype(str))
            encoders[col] = le
    else:
        for col in categorical_cols:
            if col in encoders:
                df_out[col] = df_out[col].apply(
                    lambda x: encoders[col].transform([x])[0] if x in encoders[col].classes_
                    else len(encoders[col].classes_) // 2
                )
            else:
                df_out[col] = 0

    num_cols = ['year', 'mileage', 'car_age', 'mileage_per_year']
    if fit:
        scaler = StandardScaler()
        df_out[num_cols] = scaler.fit_transform(df_out[num_cols])
    else:
        df_out[num_cols] = scaler.transform(df_out[num_cols])

    features = [
        'brand', 'model', 'year', 'mileage', 'transmission', 'fuel_type',
        'car_age', 'mileage_per_year', 'is_premium', 'is_diesel',
        'is_automatic', 'is_hybrid', 'is_suv'
    ]
    return df_out, features, encoders, scaler

def main():
    # Load data
    df = pd.read_sql_query("SELECT * FROM usedCars", source_conn)
    print(f"Vehicles: {len(df)}, Missing prices: {df['price'].isnull().sum()} "
          f"({df['price'].isnull().sum() / len(df) * 100:.1f}%)")

    # PREPROCESSING
    df_train = df[df['price'].notnull()].copy()
    df_missing = df[df['price'].isnull()].copy()

    # Outlier removal
    q05 = df_train['price'].quantile(0.05)
    q95 = df_train['price'].quantile(0.95)
    df_train = df_train[(df_train['price'] >= q05) & (df_train['price'] <= q95)]

    print(f"After cleaning: {len(df_train)} vehicles")

    # Feature engineering (fit encoders + scaler)
    df_train_fe, features, encoders, scaler = feature_engineering(df_train, fit=True)
    X = df_train_fe[features]
    y = df_train_fe['price']

    print("\nFINAL STATISTICS:")
    print(f"Vehicles: {len(X)}")
    print(f"Price - Min: {y.min():,.0f} MAD, Max: {y.max():,.0f} MAD")
    print(f"Price - Median: {y.median():,.0f} MAD, Mean: {y.mean():,.0f} MAD")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # TRAINING CATBOOST
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

    # EVALUATION
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

    # HANDLE CARS WITHOUT PRICES
    if not df_missing.empty:
        print(f"\nPredicting missing prices for {len(df_missing)} vehicles...")
        df_missing['price'] = predict_car_price_ultimate(df_missing, final_model, encoders, scaler)
        print(f"Prediction done")
    
    # # Rounding function
    # def round_to_nearest_5000(x):
    #     return int(round(x / 5000.0) * 5000)

    # df_missing['price'] = df_missing['price'].apply(round_to_nearest_5000)

    # MERGE THE DATASETS
    df_final = pd.concat([df_train, df_missing], ignore_index=True)
    df_final.to_csv("usedCars_with_predicted_prices.csv", index=False)
    print(f"Full export: {len(df_final)} saved vehicles with prices")

    # SAVE MODEL & ENCODERS
    final_model.save_model("catboost_model.cbm")
    dump(encoders, "encoders.joblib")
    dump(scaler, "scaler.joblib")

    print("\nFINAL MODEL REPORT:")
    print(f"   - {len(X):,} vehicles analyzed")
    print(f"   - R² = {r2:.3f}")
    print(f"   - Mean Absolute Error: {mae:,.0f} MAD")
    print(f"   - Accuracy: {max(0, 100 - mape):.1f}%")

if __name__ == "__main__":
    main()

