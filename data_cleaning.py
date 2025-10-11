from config import *

def main():
    # Load data from SQLite database
    df = pd.read_sql_query("SELECT * FROM usedCars", source_conn)

    print("Initial Raw Dataset")
    print(f"Initial dataset size: {len(df)} rows")

    # Remove exact duplicates across all columns
    before = len(df)
    df.drop_duplicates(keep='first', inplace=True)
    after = len(df)
    print(f"Removed {before - after} duplicate rows.")

    # Drop rows missing essential identifiers (brand or model)
    before = len(df)
    df.dropna(subset=["brand", "model"], inplace=True)
    after = len(df)
    print(f"Removed {before - after} rows with missing brand/model.")

    # Handle missing numeric values per brand (mode/median strategy)
    for brand in df["brand"].dropna().unique():
        brand_mask = (df["brand"] == brand)

        # YEAR: fill missing with mode of brand
        if df.loc[brand_mask, "year"].isna().sum() > 0:
            mode_year = df.loc[brand_mask, "year"].mode()
            if not mode_year.empty:
                df.loc[brand_mask & df["year"].isna(), "year"] = mode_year[0]

        # MILEAGE: fill missing with median of brand
        if df.loc[brand_mask, "mileage"].isna().sum() > 0:
            median_mileage = df.loc[brand_mask, "mileage"].median()
            if not np.isnan(median_mileage):
                df.loc[brand_mask & df["mileage"].isna(), "mileage"] = median_mileage

        # FUEL TYPE: fill missing with mode of brand
        if df.loc[brand_mask, "fuel_type"].isna().sum() > 0:
            mode_fuel = df.loc[brand_mask, "fuel_type"].mode()
            if not mode_fuel.empty:
                df.loc[brand_mask & df["fuel_type"].isna(), "fuel_type"] = mode_fuel[0]

    # Fill remaining missing values globally
    # Year → mode global
    if df["year"].isna().sum() > 0:
        global_year_mode = df["year"].mode()[0]
        df.loc[:, "year"] = df["year"].fillna(global_year_mode)
        print(f"Filled remaining missing 'year' with global mode: {global_year_mode}")

    # Mileage → median global
    if df["mileage"].isna().sum() > 0:
        global_mileage_median = df["mileage"].median()
        df.loc[:, "mileage"] = df["mileage"].fillna(global_mileage_median)
        print(f"Filled remaining missing 'mileage' with global median: {global_mileage_median}")

    # Fuel type → mode global
    m = df["fuel_type"].mode()
    if not m.empty:
        global_mode_fuel = m[0]
        df.loc[:, "fuel_type"] = df["fuel_type"].fillna(global_mode_fuel)
        print(f"Filled remaining missing 'fuel_type' with global mode: {global_mode_fuel}")

    # Ensure proper data types
    df["year"] = df["year"].astype(int)
    df["mileage"] = df["mileage"].astype(int)
    df["price"] = df["price"].astype(float)

    # Statistical Overview (after cleaning)
    total_cars = len(df)
    cars_with_price = df["price"].notna().sum()
    cars_without_price = df["price"].isna().sum()
    duplicates = df.duplicated(keep='first').sum()
    missing_values = df.isna().sum()

    print("\nDataset Statistics (Post-Cleaning)")
    print(f"Total cars: {total_cars}")
    print(f"Cars with price: {cars_with_price}")
    print(f"Cars without price: {cars_without_price}")
    print(f"Duplicate rows remaining: {duplicates}\n")

    print("Missing values per column:")
    print(missing_values, "\n")

    # Distribution Analysis ===
    print("Transmission Distribution")
    print(df["transmission"].value_counts(dropna=False), "\n")

    print("Fuel Type Distribution")
    print(df["fuel_type"].value_counts(dropna=False), "\n")

    # Competitive Brands and Models
    print("Top Competitive Brands and Models")
    for brand in df["brand"].dropna().unique():
        brand_df = df[df["brand"] == brand]
        model_counts = brand_df["model"].value_counts()
        total_brand = len(brand_df)
        print(f"\n{brand.upper()}")
        for model, count in model_counts.items():
            print(f"  {model} : {count}")
        print(f"Total : {total_brand}")

    # Save cleaned dataset back to DB (shared connection)
    df.to_sql(TABLE_NAME, source_conn, if_exists="replace", index=False)
    source_conn.commit()

    print("\nCleaning and Analysis Completed")
    print(f"Final dataset size after cleaning: {len(df)} rows.\n")


if __name__ == "__main__":
    main()