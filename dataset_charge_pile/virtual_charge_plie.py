import random
import sqlite3
import numpy as np

def generate_increasing_current_sequence(length,start_current,
                                         end_current):
    return np.linspace(start_current,end_current,length).tolist()

def generate_base_sequence(length, min_current, max_current):
    return [random.uniform(min_current, max_current) for _ in range(length)]


def generate_normal_charge(length, start_current, end_current):
    return np.linspace(start_current, end_current, length).tolist()

def generate_three_stage_charge(length):
    stage1 = [2000]*int(length*0.4)
    stage2 = np.linspace(2000, 500, int(length*0.3)).tolist()
    stage3 = [500]*int(length*0.3)
    return stage1 + stage2 + stage3


def add_outliers(sequence, outlier_num=5, outlier_range=(2500, 3000)):
    seq = sequence.copy()
    indices = random.sample(range(len(seq)), outlier_num)
    for idx in indices:
        seq[idx] = random.uniform(*outlier_range)
    return seq


def generate_square_wave(length, period=50, high=1200, low=800):
    return [high if i//period % 2 == 0 else low for i in range(length)]


def generate_dangerous_charge(length, start_current=1000):
    return [start_current + random.uniform(-50, 50) for _ in range(length)]

def add_gaussian_noise(sequence, mean=0, std_dev=10):
    noise = np.random.normal(mean, std_dev, len(sequence))
    return [current + noise_val for current, noise_val in zip(sequence, noise)]

def generate_current_sequence(num):
    length=1000
    if num%6==0:
         sequence=generate_base_sequence(length,min_current=800,max_current=1200)
    elif num%6==1:
         sequence=generate_normal_charge(length,start_current=2000,end_current=1200)
    elif num%6==2:
         sequence=generate_three_stage_charge(length)
    elif num%6==3:
        base_seq=generate_base_sequence(length,800,1200)
        sequence=add_outliers(sequence=base_seq)
    elif num%6==4:
        sequence=generate_square_wave(length)
    elif num%6==5:
         sequence=generate_dangerous_charge(length)

    sequence = add_gaussian_noise(sequence)
    return sequence

def create_table(connection):
    cursor = connection.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS charging_current (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            charger_id INTEGER,
            current_value REAL,
            charger_type INTEGER,
            is_disabled INTEGER DEFAULT 0  
        )
    ''')
    connection.commit()

def insert_current_sequence(connection,sequence,charger_id,charger_type):
    cursor = connection.cursor()

    data = [(charger_id, current, charger_type) for current in sequence]
    cursor.executemany('INSERT INTO charging_current (charger_id, current_value, charger_type) VALUES (?, ?, ?)', data)
    connection.commit()

def add_new_chargers(num_new_chargers):
    conn = sqlite3.connect(r'D:\Py\NET_charge\dataset_charge_pile\datas\charging_data.db')
    cursor = conn.cursor()
    for i in range(num_new_chargers):
        charger_id = random.randint(1000, 9999)
        charger_type = random.randint(0, 5)
        sequence = generate_current_sequence(charger_type)
        insert_current_sequence(conn, sequence, charger_id, charger_type)
    conn.close()

def update_charging_current_table():
    conn = sqlite3.connect(r'D:\Py\NET_charge\dataset_charge_pile\datas\charging_data.db')
    cursor = conn.cursor()
    try:
        cursor.execute('ALTER TABLE charging_current ADD COLUMN voltage REAL')
        cursor.execute('ALTER TABLE charging_current ADD COLUMN rate REAL')
        conn.commit()
    except sqlite3.OperationalError:
        pass
    conn.close()

def create_charging_record_table(connection):
    cursor = connection.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS charging_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            charger_id INTEGER,
            start_time TEXT,
            end_time TEXT,
            charging_amount REAL,
            cost REAL,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (charger_id) REFERENCES charging_current(charger_id)
        )
    ''')
    connection.commit()

def create_user_table_in_charging_data():
    conn = sqlite3.connect(r'D:\Py\NET_charge\dataset_charge_pile\datas\charging_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'user'
        )
    ''')
    conn.commit()
    conn.close()

if __name__=="__main__":
    conn = sqlite3.connect(r'D:\Py\NET_charge\dataset_charge_pile\datas\charging_data.db')
    create_table(conn)
    create_charging_record_table(conn)
    create_user_table_in_charging_data()
    for i in range(20):
        charger_id=i
        charger_type=i%6
        sequence=generate_current_sequence(i)
        insert_current_sequence(conn,sequence,charger_id,charger_type)
    conn.close()
    print("电流序列已经成功存储到数据库中")
    add_new_chargers(5)