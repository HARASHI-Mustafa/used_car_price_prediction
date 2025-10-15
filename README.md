# (: Used Car Price Prediction :)

## 1. Project Overview
This project provides an **end-to-end machine learning pipeline** for **predicting missing car prices** in online used car listings.
It integrates **data scraping**, **cleaning**, **feature engineering**, and **price prediction** using a **CatBoost Regressor** model.

The pipeline allows you to:

- Collect real used car data from an automotive marketplace,
- Clean and preprocess the dataset,
- Train and evaluate a price prediction model,
- Predict missing prices and export a complete dataset.

---

## 2. Problem Statement
Online car marketplaces often contain incomplete listings where price information is missing. This lack of pricing data reduces the reliability of market analysis, affects transparency, and prevents users from making informed comparisons.
This makes it difficult to:

- Compare listings objectively,
- Analyse the market correctly,
- Build pricing recommendation systems.

✅ Objective:
Build a robust pipeline to predict missing car prices based on attributes (Supervised Learning) such as:
- Brand
- Model
- Year
- Mileage
- Transmission
- Fuel Type

---

## 3. Methodology and Workflow

### Step 1 : Data Collection (`scraper.py`)
- Automated web scraping of car listings from an automotive marketplace.
- Extraction of structured features such as:
  - `brand`, `model`, `year`, `mileage`, `transmission`, `fuel_type`, and `price`.
- Normalises brands and cleans raw text.
- Saves data into a local SQLite database (`used_cars.db`).

**Example Output:**
```

Starting Scraping...
Page 1: 26%|█████████▏ | 10/38 [00:10<00:28, 1.00s/it]
Reached limit (10 cars). Scraping completed successfully.
Total cars saved: 10
Total records exported: 10

```

---

### Step 2 : Data Cleaning (`data_cleaning.py`)
- Removes duplicate records
- Fills missing values by brand median/mode for:
 - Year
 - Mileage
 - Fuel type
- Drops invalid records with no `brand` or `model`
- Generates statistical reports and distributions

**Example Summary:**
```

Initial dataset size: 8,448 rows
Removed 320 duplicate rows
Removed 54 rows with missing brand/model
Filled the remaining missing 'year' with global mode: 2018
Filled remaining missing 'mileage' with global median: 130,000

````

---

### Step 3 : Price Prediction (`price_prediction.py`)
- Trains a **CatBoostRegressor** model on records with known prices.
- Generates additional features:
 - car_age
 - mileage_per_year
 - is_premium, is_diesel, is_automatic, is_hybrid, is_suv
- Predicts missing prices and **rounds to the nearest 5,000 MAD**.
- Exports the final dataset to usedCars_with_predicted_prices.csv.

**Model Configuration:**
```python
model = CatBoostRegressor(
    iterations=500,
    depth=8,
    learning_rate=0.05,
    l2_leaf_reg=3,
    random_state=42,
    verbose=100,
    early_stopping_rounds=50
)
````

**Model Performance:**

```
+ R² Score: 0.836
+ Mean Absolute Error (MAE): 24,311 MAD
+ Root Mean Square Error (RMSE): 37,544 MAD
+ Mean Absolute Percentage Error (MAPE): 14.4%
+ Approximate accuracy: 85.6%
```

→ These results demonstrate strong predictive performance and confirm that a data-driven approach can accurately estimate missing vehicle prices in large datasets.

### Step 4: Testing (test_model.py)

- Loads the trained model and encoders.
- Runs test predictions on reference vehicles.

Example Predictions

| Brand         | Model    | Year | Mileage | Predicted Price (MAD) | EUR    |
| ------------- | -------- | ---- | ------- | --------------------- | ------ |
| RENAULT       | Clio     | 2020 | 40,000  | 144,174               | 13,120 |
| VOLKSWAGEN    | Golf 7   | 2018 | 60,000  | 196,281               | 17,862 |
| DACIA         | Sandero  | 2022 | 20,000  | 114,236               | 10,395 |
| MERCEDES-BENZ | Classe C | 2021 | 25,000  | 412,341               | 37,523 |
| PEUGEOT       | 3008     | 2020 | 45,000  | 256,800               | 23,369 |

---

## 4. Project Structure

used_car_price_prediction/
│
├── app.py                     # Main CLI menu
├── scraper.py                 # Data scraping
├── data_cleaning.py           # Cleaning & preprocessing
├── price_prediction.py        # Model training and prediction
├── test_model.py              # Testing & sample predictions
├── config.py                  # Global config & database connection
│
├── catboost_model.cbm         # Trained model file
├── encoders.joblib            # Encoders
├── scaler.joblib              # Scaler
│
├── used_cars.db               # SQLite database
├── usedCars_with_predicted_prices.csv
│
├── requirements.txt
├── .gitignore
└── README.md

---

## 5. Feature Importance

| Feature          | Importance (%) |
| ---------------- | -------------- |
| is_premium       | 18.47          |
| model            | 14.67          |
| car_age          | 13.71          |
| brand            | 12.86          |
| is_automatic     | 9.86           |
| year             | 9.76           |
| transmission     | 7.94           |
| mileage          | 3.49           |
| mileage_per_year | 2.76           |
| fuel_type        | 2.54           |

---

## 6. Technologies Used

* **Python 3.x**
* **Pandas**, **NumPy** – Data manipulation and processing
* **BeautifulSoup4** – HTML parsing for data extraction
* **SQLite3** – Lightweight local database
* **CatBoost** – Gradient boosting model for regression
* **Scikit-learn** – Metrics and preprocessing utilities
* **tqdm** – progress bar visualisation

---

## 7. How to Run the Project

- **Install dependencies** 
```
pip install -r requirements.txt
```

- **Launch the CLI Menu**
```
python app.py
```
- **Menu Options**
1. Create the dataset (scrape data)
2. Filter the dataset (cleaning)
3. Predict missing prices (model training & inference)
4. Test the model (sample predictions)
5. Exit

- **Output files**
 - `used_cars.db` → Raw + cleaned data
 - `catboost_model.cbm` → Trained ML model
 - `usedCars_with_predicted_prices.csv` → Final dataset with prices


- **Author** → **HARRACHI Mustapha**

📍  Data Science & AI Enthusiast

📧 harrachimustapha25@gmail.com
