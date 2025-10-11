# used_car_price_prediction

## 1. Project Overview
This project focuses on estimating missing car prices in online used car listings. The dataset collected from a public automotive marketplace contained numerous entries without price information. To address this issue, a Machine Learning model was developed to predict these missing prices based on other car attributes such as brand, model, year, mileage, transmission, and fuel type.

The workflow integrates **data scraping**, **cleaning**, **feature engineering**, and **machine learning prediction** using a **CatBoost Regressor** model. The final system provides a complete end-to-end pipeline capable of automatically retrieving, processing, and predicting car prices.

---

## 2. Problem Statement
Online car marketplaces often contain incomplete listings where price information is missing. This lack of pricing data reduces the reliability of market analysis, affects transparency, and prevents users from making informed comparisons.

The objective of this project is to:
- Collect structured data of used car listings.
- Clean and standardize this data.
- Predict the missing car prices using supervised learning.

---

## 3. Methodology and Workflow

### Step 1: Data Collection (`scraper.py`)
- Automated web scraping of car listings from an automotive marketplace.
- Extraction of structured features such as:
  - `brand`, `model`, `year`, `mileage`, `transmission`, `fuel_type`, and `price`.
- Storage of the data in a local SQLite database for reproducibility.

**Example Output:**
```

Starting Scraping...
Page 1: 26%|█████████▏ | 10/38 [00:10<00:28, 1.00s/it]
Reached limit (10 cars). Scraping completed successfully.
Total cars saved: 10
Total records exported: 10

```

---

### Step 2: Data Cleaning and Filtering (`data_cleaning.py`)
- Handling of missing or inconsistent values.
- Removal of duplicate entries and irrelevant records.
- Standardization of fuel types and transmission categories.
- Creation of derived features such as `car_age` and `mileage_per_year`.

**Example Summary:**
```

Initial records: 8448
Records with missing prices: 2907 (34.4%)
Final dataset after filtering: 4999 vehicles

````

---

### Step 3: Price Prediction (`price_prediction.py`)
- Splitting of the dataset into training and testing sets.
- Model: **CatBoostRegressor** trained on all relevant features.
- Hyperparameters tuned to balance performance and overfitting.
- Evaluation metrics calculated on the test set.

**Model Configuration:**
```python
model = CatBoostRegressor(
    iterations=500,
    depth=8,
    learning_rate=0.05,
    loss_function='RMSE',
    eval_metric='R2',
    random_seed=42,
    verbose=100
)
````

**Model Performance:**

```
R² Score: 0.836
Mean Absolute Error (MAE): 24,311 MAD
Root Mean Square Error (RMSE): 37,544 MAD
Mean Absolute Percentage Error (MAPE): 14.4%
Approximate accuracy: 85.6%
```

---

### Step 4: Feature Importance

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

### Step 5: Example Predictions

| Vehicle                     | Predicted Price (MAD) | Approx. EUR |
| --------------------------- | --------------------- | ----------- |
| RENAULT Clio 2020           | 144,174               | 13,120      |
| VOLKSWAGEN Golf 7 2018      | 196,281               | 17,862      |
| DACIA Sandero 2022          | 114,236               | 10,395      |
| MERCEDES-BENZ Classe C 2021 | 412,341               | 37,523      |
| PEUGEOT 3008 2020           | 256,800               | 23,369      |

---

## 4. Project Structure

```
used_car_price_prediction/
│
├── main.py               # Main entry point and menu interface
├── database.py           # Data scraping and database creation
├── data_cleaning.py      # Data cleaning and feature engineering
├── price_prediction.py   # Machine Learning model training and prediction
├── config.py             # Shared configuration and global settings
│
├── requirements.txt      # Dependencies
├── .gitignore            # Ignored files and folders
└── README.md             # Project documentation
```

---

## 5. Results Summary

```
Total vehicles analyzed: 4,999
R² = 0.836
MAE = 24,311 MAD
Accuracy ≈ 85.6%
Model exported as: car_price_model_ultimate.pkl
```

These results demonstrate strong predictive performance and confirm that a data-driven approach can accurately estimate missing vehicle prices in large datasets.

---

## 6. Technologies Used

* **Python 3.x**
* **Pandas**, **NumPy** – Data manipulation and processing
* **BeautifulSoup4** – HTML parsing for data extraction
* **SQLite3** – Lightweight local database
* **CatBoost** – Gradient boosting model for regression
* **Scikit-learn** – Metrics and preprocessing utilities
* **Joblib** – Model persistence
