import random
import sqlite3
import schedule
import time
from virtual_charge_plie import generate_current_sequence

def update_charger_data():
    conn = sqlite3.connect(r'D:\Py\NET_charge\dataset_charge_pile\datas\charging_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT charger_id, charger_type FROM charging_current GROUP BY charger_id")
    chargers = cursor.fetchall()

    for charger_id, charger_type in chargers:
        sequence = generate_current_sequence(charger_type)
        new_current = sequence[random.randint(0, len(sequence) - 1)]
        cursor.execute("UPDATE charging_current SET current_value =? WHERE charger_id =?",
                       (new_current, charger_id))

    conn.commit()
    conn.close()

schedule.every(5).minutes.do(update_charger_data)

while True:
    schedule.run_pending()
    time.sleep(1)