from datetime import datetime
import sqlite3

def log_operation(username, operation, details=""):
    operation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(r'D:\Py\NET_charge\dataset_charge_pile\datas\log.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO operation_logs (username, operation, operation_time, details) VALUES (?,?,?,?)",
                   (username, operation, operation_time, details))
    conn.commit()
    conn.close()