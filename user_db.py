import sqlite3
import bcrypt
from log import log_operation

def create_user_table():
    conn = sqlite3.connect(r'D:\Py\NET_charge\dataset_charge_pile\datas\users.db')
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

def insert_user(username, password, role='user'):
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    conn1 = sqlite3.connect(r'D:\Py\NET_charge\dataset_charge_pile\datas\users.db')
    cursor1 = conn1.cursor()
    try:
        cursor1.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", (username, hashed, role))
        conn1.commit()
    except sqlite3.IntegrityError:
        print("用户名已存在，请选择其他用户名。")
        conn1.close()
        return
    conn1.close()


    conn2 = sqlite3.connect(r'D:\Py\NET_charge\dataset_charge_pile\datas\charging_data.db')
    cursor2 = conn2.cursor()
    try:
        cursor2.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", (username, hashed, role))
        conn2.commit()
        log_operation(username, "用户注册", f"角色: {role}")
    except sqlite3.IntegrityError:
        print("用户名已存在，请选择其他用户名。")
    finally:
        conn2.close()

def check_user(username, password):
    conn = sqlite3.connect(r'D:\Py\NET_charge\dataset_charge_pile\datas\users.db')
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE username =?", (username,))
    user = cursor.fetchone()
    conn.close()
    if user:
        hashed = user[0]
        return bcrypt.checkpw(password.encode('utf-8'), hashed)
    return False

def get_user_role(username):
    conn = sqlite3.connect(r'D:\Py\NET_charge\dataset_charge_pile\datas\users.db')
    cursor = conn.cursor()
    cursor.execute("SELECT role FROM users WHERE username =?", (username,))
    role = cursor.fetchone()
    conn.close()
    return role[0] if role else 'user'

def create_reservation_table():
    conn = sqlite3.connect(r'D:\Py\NET_charge\dataset_charge_pile\datas\reservation.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reservations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            charger_id INTEGER,
            reservation_time TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (charger_id) REFERENCES charging_current(charger_id)
        )
    ''')
    conn.commit()
    conn.close()


def cancel_reservation(user_id, charger_id):
    conn = sqlite3.connect(r'D:\Py\NET_charge\dataset_charge_pile\datas\reservation.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM reservations WHERE user_id =? AND charger_id =?", (user_id, charger_id))
    conn.commit()
    conn.close()

def update_user(username, new_username=None, new_password=None):
    if new_username:
        new_username_hashed = new_username
    if new_password:
        new_password_hashed = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())


    conn1 = sqlite3.connect(r'D:\Py\NET_charge\dataset_charge_pile\datas\users.db')
    cursor1 = conn1.cursor()
    if new_username:
        try:
            cursor1.execute("UPDATE users SET username =? WHERE username =?", (new_username, username))
        except sqlite3.IntegrityError:
            print("用户名已存在，请选择其他用户名。")
            conn1.close()
            return
    if new_password:
        cursor1.execute("UPDATE users SET password =? WHERE username =?", (new_password_hashed, username))
    conn1.commit()
    conn1.close()


    conn2 = sqlite3.connect(r'D:\Py\NET_charge\dataset_charge_pile\datas\charging_data.db')
    cursor2 = conn2.cursor()
    if new_username:
        try:
            cursor2.execute("UPDATE users SET username =? WHERE username =?", (new_username, username))
        except sqlite3.IntegrityError:
            print("用户名已存在，请选择其他用户名。")
    if new_password:
        cursor2.execute("UPDATE users SET password =? WHERE username =?", (new_password_hashed, username))
    conn2.commit()
    conn2.close()

if __name__ == "__main__":
    create_user_table()
    create_reservation_table()