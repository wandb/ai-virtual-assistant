# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import re
import psycopg2
from datetime import datetime

import argparse

# Set up the argument parser
parser = argparse.ArgumentParser(description='Database connection parameters.')
parser.add_argument('--dbname', type=str, default='customer_data', help='Database name')
parser.add_argument('--user', type=str, default='postgres', help='Database user')
parser.add_argument('--password', type=str, default='password', help='Database password')
parser.add_argument('--host', type=str, default='localhost', help='Database host')
parser.add_argument('--port', type=str, default='5432', help='Database port')

# Parse the arguments
args = parser.parse_args()

# Database connection parameters
db_params = {
    'dbname': args.dbname,
    'user': args.user,
    'password': args.password,
    'host': args.host,
    'port': args.port
}

# CSV file path
csv_file_path = './data/orders.csv'

# Connect to the database
conn = psycopg2.connect(**db_params)
cur = conn.cursor()

# Drop the table if it exists
drop_table_query = '''
DROP TABLE IF EXISTS customer_data;
'''

# Create the table if it doesn't exist
create_table_query = '''
CREATE TABLE IF NOT EXISTS customer_data (
    customer_id INTEGER NOT NULL,
    order_id INTEGER NOT NULL,
    product_name VARCHAR(255) NOT NULL,
    product_description VARCHAR NOT NULL,
    order_date DATE NOT NULL,
    quantity INTEGER NOT NULL,
    order_amount DECIMAL(10, 2) NOT NULL,
    order_status VARCHAR(50),
    return_status VARCHAR(50),
    return_start_date DATE,
    return_received_date DATE,
    return_completed_date DATE,
    return_reason VARCHAR(255),
    notes TEXT,
    PRIMARY KEY (customer_id, order_id)
);
'''
cur.execute(drop_table_query)
cur.execute(create_table_query)

# Open the CSV file and insert data
with open(csv_file_path, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip the header row

    for row in reader:
        # Access columns by index as per the provided structure
        order_id = int(row[1])  # OrderID
        customer_id = int(row[0])  # CID (Customer ID)

        # Correcting the order date to include time
        order_date = datetime.strptime(row[4], "%Y-%m-%dT%H:%M:%S")  # OrderDate with time

        quantity = int(row[5])  # Quantity

        # Handle optional date fields with time parsing
        return_start_date = datetime.strptime(row[9], "%Y-%m-%dT%H:%M:%S") if row[9] else None  # ReturnStartDate
        return_received_date = datetime.strptime(row[10],"%Y-%m-%dT%H:%M:%S") if row[10] else None  # ReturnReceivedDate
        return_completed_date = datetime.strptime(row[11], "%Y-%m-%dT%H:%M:%S") if row[11] else None  # ReturnCompletedDate

        # Clean product name
        product_name = re.sub(r'[®™]', '', row[2])  # ProductName

        product_description = re.sub(r'[®™]', '', row[3])
        # OrderAmount as float
        order_amount = float(row[6].replace(',', ''))

        # Insert data into the database
        cur.execute(
            '''
            INSERT INTO customer_data (
                customer_id, order_id, product_name, product_description, order_date, quantity, order_amount,
                order_status, return_status, return_start_date, return_received_date,
                return_completed_date, return_reason, notes
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''',
            (customer_id, order_id, product_name, product_description, order_date, quantity, order_amount,
             row[7],  # OrderStatus
             row[8],  # ReturnStatus
             return_start_date, return_received_date, return_completed_date,
             row[12],  # ReturnReason
             row[13])  # Notes
        )

# Commit the changes and close the connection
conn.commit()
cur.close()
conn.close()

print("CSV Data imported successfully!")
