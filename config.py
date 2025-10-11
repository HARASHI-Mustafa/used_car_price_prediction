# LIBRARIES
import os
import re
import difflib
import warnings
import sqlite3
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings('ignore')

# DATABASE SETTINGS
DB_NAME = "used_cars.db"
TABLE_NAME = "usedCars"

# Create connection
source_conn = sqlite3.connect(DB_NAME)
source_cursor = source_conn.cursor()
