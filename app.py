from flask import Flask, render_template, send_from_directory, session, redirect, url_for
from views import maintenance_bp, get_faulty_chargers,get_user_id
import pandas as pd
import sqlite3
from datetime import datetime
from user_db import create_user_table, create_reservation_table

app = Flask(__name__)
app.register_blueprint(maintenance_bp, url_prefix='/maintenance')

def get_reserved_chargers():
    try:
        conn = sqlite3.connect(r'dataset_charge_pile\datas\reservation.db')
        query = "SELECT charger_id FROM reservations"
        data = pd.read_sql_query(query, conn)
        conn.close()
        charger_ids = data['charger_id'].tolist()
        total_reserved = len(charger_ids)
        return total_reserved, charger_ids
    except Exception as e:
        print(f"Error reading data: {e}")
        return 0, []

@app.route('/image/<path:filename>')
def image(filename):
    return send_from_directory('static', filename)

@app.route('/elements/<path:filename>')
def frame(filename):
    return send_from_directory('static', filename)

def get_charge_pile_data():
    try:
        conn = sqlite3.connect(r'dataset_charge_pile\datas\charging_data.db')

        query = "SELECT COUNT(DISTINCT charger_id) FROM charging_current"
        data = pd.read_sql_query(query, conn)
        total_charging_piles = data.iloc[0, 0]
        conn.close()

        faulty_chargers = get_faulty_chargers()
        alarm_count = len(faulty_chargers)
        return total_charging_piles, alarm_count
    except Exception as e:
        print(f"Error reading data: {e}")
        return 0, 0

def get_last_7_days_charging_piles():
    try:
        conn = sqlite3.connect(r'dataset_charge_pile\datas\last_7_used_data.db')
        query = "SELECT date, used_counts, alarm_counts FROM charging_current ORDER BY date ASC"
        data = pd.read_sql_query(query, conn)
        conn.close()
        dates = data['date'].tolist()
        used_counts = data['used_counts'].tolist()
        alarm_counts = data['alarm_counts'].tolist()
        return dates, used_counts, alarm_counts
    except Exception as e:
        print(f"Error reading data: {e}")
        return [], []

def get_disabled_chargers_count():
    try:
        conn = sqlite3.connect(r'dataset_charge_pile\datas\charging_data.db')
        query = "SELECT COUNT(DISTINCT charger_id) FROM charging_current WHERE is_disabled = 1"
        data = pd.read_sql_query(query, conn)
        conn.close()
        return data.iloc[0, 0]
    except Exception as e:
        print(f"Error reading data: {e}")
        return 0

def get_charging_records(user_id=None):
    conn = sqlite3.connect(r'dataset_charge_pile\datas\charging_data.db')
    if user_id:
        query = f"SELECT * FROM charging_records WHERE user_id = {user_id}"
    else:
        query = "SELECT * FROM charging_records"
    data = pd.read_sql_query(query, conn)
    conn.close()
    return data.to_dict('records')

@app.route('/')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('maintenance.login'))
    total_charging_piles, alarm_count = get_charge_pile_data()
    charging_dates, charging_7_counts, alarm_7_counts = get_last_7_days_charging_piles()
    faulty_chargers = get_faulty_chargers()
    disabled_chargers_count = get_disabled_chargers_count()
    total_reserved, reserved_charger_ids = get_reserved_chargers()
    user_id = get_user_id(session['username'])
    charging_records = get_charging_records(user_id)
    return render_template('index.html', total_charging_piles=total_charging_piles, alarm_count=alarm_count,
                           charging_dates=charging_dates, charging_counts=charging_7_counts,
                           alarm_dates=charging_dates, alarm_counts=alarm_7_counts, faulty_chargers=faulty_chargers,
                           disabled_chargers_count=disabled_chargers_count,
                           total_reserved=total_reserved, reserved_charger_ids=reserved_charger_ids,
                           charging_records=charging_records)
@app.route('/system_introduction')
def system_introduction():
    if 'username' not in session:
        return redirect(url_for('maintenance.login'))
    return render_template('system_introduction.html')


def update_charging_current_table():
    conn = sqlite3.connect(r'dataset_charge_pile\datas\charging_data.db')
    cursor = conn.cursor()
    try:
        cursor.execute('ALTER TABLE charging_current ADD COLUMN voltage REAL')
        cursor.execute('ALTER TABLE charging_current ADD COLUMN rate REAL')
        conn.commit()
    except sqlite3.OperationalError:
        pass
    conn.close()

def create_repair_tables():
    conn = sqlite3.connect(r'dataset_charge_pile\datas\charging_data.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS repair_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            charger_id INTEGER,
            request_time DATETIME,
            description TEXT,
            status TEXT DEFAULT 'pending'
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS repair_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id INTEGER,
            admin_id INTEGER,
            response_time DATETIME,
            response_text TEXT,
            FOREIGN KEY (request_id) REFERENCES repair_requests(id)
        )
    ''')
    conn.commit()
    conn.close()

def create_log_table():
    conn = sqlite3.connect(r'dataset_charge_pile\datas\log.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS operation_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            operation TEXT,
            operation_time DATETIME,
            details TEXT
        )
    ''')
    conn.commit()
    conn.close()

def create_recharge_tables():
    conn = sqlite3.connect(r'dataset_charge_pile\datas\charging_data.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recharge_qrcodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            charger_id INTEGER,
            qrcode_path TEXT,
            upload_time DATETIME,
            FOREIGN KEY (charger_id) REFERENCES charging_current(charger_id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recharge_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            charger_id INTEGER,
            recharge_time DATETIME,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (charger_id) REFERENCES charging_current(charger_id)
        )
    ''')
    conn.commit()
    conn.close()

def create_feedback_table():
    conn = sqlite3.connect(r'dataset_charge_pile\datas\charging_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            feedback_type TEXT, 
            content TEXT,
            submit_time DATETIME,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    conn.commit()
    conn.close()


if __name__ == '__main__':
    create_user_table()
    create_reservation_table()
    update_charging_current_table()
    create_repair_tables()
    create_log_table()
    create_recharge_tables()
    create_feedback_table()
    app.secret_key = 'your_secret_key'
    app.run(host='0.0.0.0', port=5000, debug=True)
