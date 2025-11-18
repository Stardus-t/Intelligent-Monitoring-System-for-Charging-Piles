import random
import sqlite3
import numpy as np

def create_table(connection):
    cursor=connection.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS charging_current (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date INTEGER,
            used_counts REAL,
            alarm_counts REAL
        )
    ''')
    connection.commit()

def insert_current_sequence(connection,date):
    cursor = connection.cursor()
    used_counts = random.randint(0,21)
    alarm_counts=random.randint(0,used_counts)
    cursor.execute('INSERT INTO charging_current (date, used_counts, alarm_counts) VALUES (?, ?, ?)',
                   (date,used_counts,alarm_counts))
    connection.commit()



if __name__=="__main__":
    conn = sqlite3.connect(r'D:\Py\NET_charge\dataset_charge_pile\datas\last_7_used_data.db')
    create_table(conn)
    for i in range(1,8):
        date=i
        insert_current_sequence(conn,date)
    conn.close()
    print("电流序列已经成功存储到数据库中")