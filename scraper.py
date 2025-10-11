from config import *

# CONFIGURATION
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/91.0.4472.124 Safari/537.36'
}

def create_table():
    source_cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            brand TEXT,
            model TEXT,
            year INTEGER,
            mileage INTEGER,
            transmission TEXT,
            fuel_type TEXT,
            price REAL,
            UNIQUE(brand, model, year, mileage, transmission, fuel_type)
        )
    """)
    source_conn.commit()


def insert_record(data):
    source_cursor.execute(f"""
        INSERT OR IGNORE INTO {TABLE_NAME}
        (brand, model, year, mileage, transmission, fuel_type, price)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, data)
    source_conn.commit()


def export_cleaned_dataset():
    # Load data from source database
    df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", source_conn)

    # Clean and transform the data
    df.dropna(subset=['brand', 'model', 'year'], inplace=True)
    df.drop_duplicates(inplace=True)

    # Ensure correct data types
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')

    # Reorder columns with price at the end
    cols = ['brand', 'model', 'year', 'mileage', 'transmission', 'fuel_type', 'price']
    df = df[cols]

    print(f"Total records exported: {len(df)}")
    return df


# NETWORK FUNCTIONS
def get_page(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        print(f"ERROR Failed to access {url} - {e}")
        return None


# BRAND NORMALIZATION
def normalize_brand(brand_name):
    valid_brands = [
        "ABARTH", "ACURA", "ALFA ROMEO", "ALPINE", "ASTON MARTIN", "AUDI", "BENTLEY",
        "BMW", "BYD", "CADILLAC", "CHERY", "CHEVROLET", "CHRYSLER", "CITROEN", "CUPRA",
        "DACIA", "DAEWOO", "DAIHATSU", "DFSK", "DODGE", "DS", "FERRARI", "FIAT", "FORD",
        "FOTON", "FUSO", "GAZELLE", "GEELY", "GMC", "GREAT WALL", "HINO", "HONDA", "HUMMER",
        "HYUNDAI", "INFINITI", "ISUZU", "IVECO", "JAGUAR", "JEEP", "KIA", "LANCIA",
        "LAND-ROVER", "LEXUS", "LINCOLN", "MAHINDRA", "MASERATI", "MAZDA", "MAZINO",
        "MERCEDES-BENZ", "MG", "MINI", "MITSUBISHI", "NISSAN", "OPEL", "PEUGEOT", "PORSCHE",
        "RENAULT", "SEAT", "SERES", "SKODA", "SMART", "SSANGYONG", "SUBARU", "SUZUKI",
        "TATA", "TESLA", "TOYOTA", "VOLKSWAGEN", "VOLVO"
    ]

    if not brand_name:
        return None

    brand_upper = brand_name.upper()
    if brand_upper in valid_brands:
        return brand_upper

    closest = difflib.get_close_matches(brand_upper, valid_brands, n=1, cutoff=0.8)
    return closest[0] if closest else brand_upper


# SCRAPING LOGIC
def extract_car_data(url):
    page = get_page(url)
    if not page:
        return None

    soup = BeautifulSoup(page.content, "html.parser")
    data = {}

    # Extract key/value pairs
    for div in soup.find_all('div', class_='sc-19cngu6-1 doRGIC'):
        spans = div.find_all('span')
        if len(spans) == 2:
            value, label = spans[0].text.strip(), spans[1].text.strip()
            data[label] = value

    # Map data
    brand = normalize_brand(data.get("Marque"))
    model = data.get("Modèle")
    year = data.get("Année-Modèle")
    mileage = data.get("Kilométrage")
    transmission = data.get("Boite de vitesses")
    fuel_type = data.get("Type de carburant")

    # Clean numeric values
    year = int(re.sub(r'\D', '', year)) if year else None
    mileage = int(re.findall(r'\d[\d ]*', mileage)[-1].replace(' ', '')) if mileage else None

    return brand, model, year, mileage, transmission, fuel_type

# MAIN SCRAPER
def main(base_url):
    create_table()
    total = 0
    page_number = 1
    MAX_CARS = 10

    print("\nStarting Scraping...")

    while total < MAX_CARS:
        url = f"{base_url}?o={page_number}"
        response = get_page(url)
        if not response:
            break

        soup = BeautifulSoup(response.content, "lxml")
        car_links = soup.find_all("a", href=lambda x: x and "/voitures_d_occasion/" in x)
        if not car_links:
            print("No more listings found. Scraping complete.")
            break

        for car_element in tqdm(car_links, desc=f"Page {page_number}", ncols=80):
            if total >= MAX_CARS:
                print(f"Reached limit ({total} cars).")
                break

            # Extract title and price
            title_tag = car_element.find('p', attrs={'title': True})
            title = title_tag.get('title').strip() if title_tag else "Unknown"

            price_tag = car_element.find('span', class_='sc-b88r7z-2 eVweKh')
            if price_tag:
                raw_price = price_tag.text.strip()
                price = float(re.sub(r'[^0-9]', '', raw_price)) if raw_price else None
            else:
                price = None

            # Extract full car details
            car_data = extract_car_data(car_element['href'])
            if not car_data:
                continue

            brand, model, year, mileage, transmission, fuel_type = car_data

            # Insert into database
            insert_record((brand, model, year, mileage, transmission, fuel_type, price))
            total += 1

        page_number += 1

    print(f"\nScraping completed successfully. Total cars saved: {total}")

    # Export cleaned dataset after scraping
    cleaned_df = export_cleaned_dataset()
    print(cleaned_df.head())
    source_conn.close()

if __name__ == "__main__":
    main("https://www.avito.ma/fr/maroc/voitures_d_occasion-à_vendre")