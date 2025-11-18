import sqlite3
import pandas as pd

def get_charging_records_for_report():
    try:
        conn = sqlite3.connect(r'D:\Py\NET_charge\dataset_charge_pile\datas\charging_data.db')
        query = "SELECT * FROM charging_records"
        data = pd.read_sql_query(query, conn)
        conn.close()
        return data.to_dict('records')
    except Exception as e:
        print(f"Error reading charging records: {e}")
        return []

def get_repair_requests_for_report():
    try:
        conn = sqlite3.connect(r'D:\Py\NET_charge\dataset_charge_pile\datas\charging_data.db')
        query = "SELECT * FROM repair_requests"
        data = pd.read_sql_query(query, conn)
        conn.close()
        return data.to_dict('records')
    except Exception as e:
        print(f"Error reading repair requests: {e}")
        return []