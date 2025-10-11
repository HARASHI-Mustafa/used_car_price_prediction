from config import *
from scraper import main as first_code
from data_cleaning import main as second_code
from price_prediction import main as third_code

def menu():
    database_created = False
    if os.path.isfile(DB_NAME):
        try:
            source_cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
            count = source_cursor.fetchone()[0]
            if count >= 10:
                database_created = True
            else:
                print(f"Database exists but table '{TABLE_NAME}' has less than 10 rows ({count}).")
        except sqlite3.OperationalError:
            print(f"Table '{TABLE_NAME}' does not exist. Please create the dataset first.")
    else:
        print(f"Database file '{DB_NAME}' does not exist. Please create the dataset first.")

    try:
        while True:
            print()
            print("1. Create the dataset")
            print("2. Filter the dataset")
            print("3. Predict missing prices")
            print("4. Exit")
            print()

            choice = int(input("Select an option (1-4): "))

            if choice == 1:
                first_code('https://www.avito.ma/fr/maroc/voitures_d_occasion-%C3%A0_vendre')
                database_created = True
            elif choice == 2:
                if not database_created:
                    print("The database does not exist. Please create it first.")
                else:
                    second_code()
            elif choice == 3:
                if not database_created:
                    print("The database does not exist. Please create it first.")
                else:
                    third_code()
            elif choice == 4:
                print("Goodbye!")
                break
            else:
                print("Invalid option. Please select a valid one.")
    except KeyboardInterrupt:
        print()
        print("Program manually interrupted. Goodbye!")

# Check if the used_cars.db file exists
if os.path.isfile("used_cars.db"):
    database_created = True

# Call the main menu
if __name__ == "__main__":
    menu()