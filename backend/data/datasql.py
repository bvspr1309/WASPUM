import os
import pandas as pd
from sqlalchemy import create_engine

# MySQL server connection details
host = 'localhost'
user = 'DBadmin'
password = 'root13Musashi'
database = 'waspum'

# Create a MySQL database connection
engine = create_engine(f'mysql://{user}:{password}@{host}/{database}')

# Folder containing CSV files
csv_folder = r'D:\Files\Projects\Capstone\Code\Data'

# Loop through CSV files and import them into tables
for filename in os.listdir(csv_folder):
    if filename.endswith(".csv"):
        table_name = os.path.splitext(filename)[0]
        df = pd.read_csv(os.path.join(csv_folder, filename))
        df.to_sql(table_name, con=engine, if_exists='replace', index=False, schema=database)

# Close the database connection
engine.dispose()
