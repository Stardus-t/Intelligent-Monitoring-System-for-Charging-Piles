import os
import sqlite3
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import pandas as pd
import bcrypt
from report import get_repair_requests_for_report,get_charging_records_for_report
from log import log_operation
from datetime import datetime
from user_db import insert_user,get_user_role,cancel_reservation
from flask import Flask, render_template,  Blueprint, jsonify, request, redirect, url_for, session

app = Flask(__name__, template_folder='templates')
maintenance_bp = Blueprint('maintenance', __name__)
app.secret_key = 'key'

@maintenance_bp.route('/add_charger', methods=['GET', 'POST'])
def add_charger():
    if 'username' not in session or ('role' not in session or session['role'] != 'admin'):
        return redirect(url_for('maintenance.login'))
    if request.method == 'POST':
        charger_id = request.form.get('charger_id')
        current = request.form.get('current')
        voltage = request.form.get('voltage')
        rate = request.form.get('rate')
        charger_type = request.form.get('charger_type')
        try:
            conn = sqlite3.connect(r'dataset_charge_pile\datas\charging_data.db')
            cursor = conn.cursor()
            cursor.execute("INSERT INTO charging_current (charger_id, current_value, voltage, rate, charger_type) VALUES (?,?,?,?,?)",
                           (charger_id, current, voltage, rate, charger_type))
            conn.commit()
            conn.close()
            log_operation(session['username'], f"添加充电桩，ID: {charger_id}",
                          f"电流: {current}, 电压: {voltage}, 费率: {rate}, 类型: {charger_type}")
            return redirect(url_for('maintenance.charger_details'))
        except Exception as e:
            return f"添加失败: {str(e)}"
    return render_template('add_charger.html')

@maintenance_bp.route('/edit_charger/<int:charger_id>', methods=['GET', 'POST'])
def edit_charger(charger_id):
    if 'username' not in session or ('role' not in session or session['role'] != 'admin'):
        return redirect(url_for('maintenance.login'))
    if request.method == 'POST':
        current = request.form.get('current')
        voltage = request.form.get('voltage')
        rate = request.form.get('rate')
        charger_type = request.form.get('charger_type')
        try:
            conn = sqlite3.connect(r'dataset_charge_pile\datas\charging_data.db')
            cursor = conn.cursor()
            cursor.execute("UPDATE charging_current SET current_value =?, voltage =?, rate =?, charger_type =? WHERE charger_id =?",
                           (current, voltage, rate, charger_type, charger_id))
            conn.commit()
            conn.close()
            return redirect(url_for('maintenance.charger_details'))
        except Exception as e:
            return f"编辑失败: {str(e)}"
    conn = sqlite3.connect(r'dataset_charge_pile\datas\charging_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT current_value, voltage, rate, charger_type FROM charging_current WHERE charger_id =?", (charger_id,))
    data = cursor.fetchone()
    conn.close()
    if data:
        current, voltage, rate, charger_type = data
        return render_template('edit_charger.html', charger_id=charger_id, current=current, voltage=voltage, rate=rate, charger_type=charger_type)
    return "未找到该充电桩"

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=200, num_classes=6, target_length=20):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc_class = nn.Linear(hidden_size, num_classes)
        self.fc_reg = nn.Linear(hidden_size, target_length)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        class_output = self.fc_class(last_output)
        reg_output = self.fc_reg(last_output)
        return class_output, reg_output

def load_model():
    model = LSTM()
    weight_path = r'weight'
    weight_file = os.path.join(weight_path, 'weight_trained.pth')
    checkpoint = torch.load(weight_file, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

def get_charger_data(charger_id):
    conn = sqlite3.connect(r'dataset_charge_pile\datas\charging_data.db')
    query = f"""
        SELECT current_value
        FROM charging_current
        WHERE charger_id={charger_id}
        ORDER BY id ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df['current_value'].values

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.reshape(-1, 1)).squeeze()
    return scaler, data_scaled

def create_sequences(data, look_back=800):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back])
        y.append(data[i+look_back])
    return np.array(X), np.array(y)

def is_unsafe_charge(predictions):
    threshold = 1000
    return any(p > threshold for p in predictions)

def get_faulty_chargers():
    conn = sqlite3.connect(r'dataset_charge_pile\datas\charging_data.db')
    query = """
        SELECT DISTINCT charger_id
        FROM charging_current
        WHERE current_value > 2000  
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df['charger_id'].tolist()

def label_to_class(label):
    class_mapping = {
        0: "基础随机序列",
        1: "正常充电（带下降趋势）",
        2: "三段式充电",
        3: "含离群点的充电",
        4: "规律性方波",
        5: "危险充电（电流无衰减）"
    }
    return class_mapping.get(label, "未知类别")

@maintenance_bp.route('/update_profile', methods=['GET', 'POST'])
def update_profile():
    if 'username' not in session:
        return redirect(url_for('maintenance.login'))
    if request.method == 'POST':
        new_username = request.form.get('new_username')
        new_password = request.form.get('new_password')
        username = session['username']
        from user_db import update_user
        update_user(username, new_username, new_password)
        if new_username:
            session['username'] = new_username
        return redirect(url_for('dashboard'))
    return render_template('update_profile.html')

@maintenance_bp.route('/enable_charger/<int:charger_id>', methods=['GET'])
def enable_charger(charger_id):
    if 'username' not in session or ('role' not in session or session['role'] != 'admin'):
        return jsonify({'success': False, 'message': '权限不足'})
    try:
        conn = sqlite3.connect(r'dataset_charge_pile\datas\charging_data.db')
        cursor = conn.cursor()
        cursor.execute("UPDATE charging_current SET is_disabled = 0 WHERE charger_id =?", (charger_id,))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': '充电桩已启用'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@maintenance_bp.route('/disable_charger/<int:charger_id>', methods=['GET'])
def disable_charger(charger_id):
    if 'username' not in session or ('role' not in session or session['role'] != 'admin'):
        return jsonify({'success': False, 'message': '权限不足'})
    try:
        conn = sqlite3.connect(r'dataset_charge_pile\datas\charging_data.db')
        cursor = conn.cursor()
        cursor.execute("UPDATE charging_current SET is_disabled = 1 WHERE charger_id =?", (charger_id,))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': '充电桩已禁用'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@maintenance_bp.route('/charger_current/<int:charger_id>', methods=['GET'])
def charger_current(charger_id):
    model = load_model()
    data = get_charger_data(charger_id)
    if len(data) < 10:
        return jsonify({'error': '数据不足'})

    scaler, data_scaled = preprocess_data(data)
    look_back = 800
    X, y = create_sequences(data_scaled, look_back)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    last_input = X_tensor[-1].unsqueeze(0)
    future_predictions = []
    prediction_steps = 20
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    last_input = last_input.to(device)
    model = model.to(device)

    with torch.no_grad():
        for _ in range(prediction_steps):
            class_output, reg_output = model(last_input)
            predicted_value = reg_output[:, 0].item()
            future_predictions.append(predicted_value)
            _, predicted_class = torch.max(class_output.data, 1)
            new_input = reg_output[:, 0].unsqueeze(1).unsqueeze(2)
            last_input = torch.cat((last_input[:, 1:, :], new_input), dim=1)

    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions_inverse = scaler.inverse_transform(future_predictions)
    data_inverse = scaler.inverse_transform(data_scaled.reshape(-1, 1))

    predicted_class_name = label_to_class(predicted_class.item())

    return jsonify({
        'original_current': data_inverse.flatten().tolist(),
        'predicted_current': future_predictions_inverse.flatten().tolist(),
        'predicted_class': predicted_class_name
    })

@maintenance_bp.route('/charger_details', methods=['GET'])
@maintenance_bp.route('/charger_details', methods=['GET'])
def charger_details():
    if 'username' not in session:
        return redirect(url_for('login'))
    model = load_model()

    conn = sqlite3.connect(r'dataset_charge_pile\datas\charging_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT charger_id FROM charging_current")
    charger_ids = [row[0] for row in cursor.fetchall()]
    conn.close()

    charger_status = {}
    faulty_chargers = get_faulty_chargers()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    reservation_conn = sqlite3.connect(r'dataset_charge_pile\datas\reservation.db')
    reservation_cursor = reservation_conn.cursor()

    for charger_id in charger_ids:
        data = get_charger_data(charger_id)
        if len(data) < 10:
            charger_status[charger_id] = {'status': '数据不足', 'predicted_class': '未知', 'is_disabled': 0, 'is_reserved': False}
            continue

        scaler, data_scaled = preprocess_data(data)
        look_back = 800
        X, y = create_sequences(data_scaled, look_back)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        last_input = X_tensor[-1].unsqueeze(0)
        last_input = last_input.to(device)
        future_predictions = []
        prediction_steps = 20
        y_inverse = scaler.inverse_transform(y.reshape(-1, 1))

        with torch.no_grad():
            for _ in range(prediction_steps):
                class_output, reg_output = model(last_input)
                predicted_value = reg_output[:, 0].item()
                future_predictions.append(predicted_value)
                _, predicted_class = torch.max(class_output.data, 1)
                new_input = reg_output[:, 0].unsqueeze(1).unsqueeze(2)
                last_input = torch.cat((last_input[:, 1:, :], new_input), dim=1)

            future_predictions = np.array(future_predictions).reshape(-1, 1)
            future_predictions_inverse = scaler.inverse_transform(future_predictions)

            if is_unsafe_charge(future_predictions_inverse.flatten()):
                charger_status[charger_id] = {'status': 'alert', 'predicted_class': label_to_class(predicted_class.item())}
            else:
                charger_status[charger_id] = {'status': 'safe', 'predicted_class': label_to_class(predicted_class.item())}

        conn = sqlite3.connect(r'dataset_charge_pile\datas\charging_data.db')
        cursor = conn.cursor()
        cursor.execute("SELECT is_disabled FROM charging_current WHERE charger_id =? LIMIT 1", (charger_id,))
        result = cursor.fetchone()
        charger_status[charger_id]['is_disabled'] = result[0] if result else 0


        user_id = 1
        reservation_cursor.execute("SELECT id FROM reservations WHERE user_id =? AND charger_id =?", (user_id, charger_id))
        reservation_result = reservation_cursor.fetchone()
        charger_status[charger_id]['is_reserved'] = bool(reservation_result)

    reservation_conn.close()

    return render_template('charger_details.html', charger_ids=charger_ids, charger_status=charger_status)
@maintenance_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']
        insert_user(username, password, role)
        return redirect(url_for('maintenance.login'))
    return render_template('register.html')

@maintenance_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect(r'dataset_charge_pile\datas\users.db')
        cursor = conn.cursor()
        cursor.execute("SELECT password, role FROM users WHERE username =?", (username,))
        result = cursor.fetchone()
        conn.close()
        if result:
            hashed = result[0]
            role = result[1]
            if bcrypt.checkpw(password.encode('utf-8'), hashed):
                session['username'] = username
                session['role'] = role
                return redirect(url_for('dashboard'))
    return render_template('login.html')

@maintenance_bp.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('maintenance.login'))


@maintenance_bp.route('/reserve_charger/<int:charger_id>', methods=['GET'])
def reserve_charger(charger_id):
    if 'username' not in session:
        return jsonify({'success': False, 'message': '请先登录'})
    username = session['username']
    role = get_user_role(username)
    if role not in ['user', 'admin']:
        return jsonify({'success': False, 'message': '权限不足'})


    conn = sqlite3.connect(r'dataset_charge_pile\datas\users.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username =?", (username,))
    user = cursor.fetchone()
    if not user:
        conn.close()
        return jsonify({'success': False, 'message': '用户不存在'})
    user_id = user[0]


    reservation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    try:
        conn_reservation = sqlite3.connect(r'dataset_charge_pile\datas\reservation.db')
        cursor_reservation = conn_reservation.cursor()
        cursor_reservation.execute("INSERT INTO reservations (user_id, charger_id, reservation_time) VALUES (?, ?, ?)",
                                   (user_id, charger_id, reservation_time))
        conn_reservation.commit()
        conn_reservation.close()
        return jsonify({'success': True, 'message': '预约成功'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})
    finally:
        conn.close()

@maintenance_bp.route('/cancel_reservation/<int:charger_id>', methods=['GET'])
def cancel_reservation_view(charger_id):
    if 'username' not in session:
        return jsonify({'success': False, 'message': '未登录'})

    user_id = 1
    try:
        cancel_reservation(user_id, charger_id)
        return jsonify({'success': True, 'message': '预约已取消'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

def get_user_id(username):
    conn = sqlite3.connect(r'dataset_charge_pile\datas\users.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username =?", (username,))
    user = cursor.fetchone()
    conn.close()
    return user[0] if user else None

def insert_charging_record(user_id, charger_id, start_time, end_time, charging_amount, cost):
    conn = sqlite3.connect(r'dataset_charge_pile\datas\charging_data.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO charging_records (user_id, charger_id, start_time, end_time, charging_amount, cost) VALUES (?, ?, ?, ?, ?, ?)',
                   (user_id, charger_id, start_time, end_time, charging_amount, cost))
    conn.commit()
    conn.close()

@app.route('/start_charge', methods=['POST'])
def start_charge():
    if 'username' not in session:
        return redirect(url_for('maintenance.login'))
    user_id = get_user_id(session['username'])
    charger_id = request.form.get('charger_id')
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    return jsonify({'success': True, 'message': '充电开始', 'start_time': start_time})

@app.route('/end_charge', methods=['POST'])
def end_charge():
    if 'username' not in session:
        return redirect(url_for('maintenance.login'))
    user_id = get_user_id(session['username'])
    charger_id = request.form.get('charger_id')
    start_time = request.form.get('start_time')
    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    charging_amount = 10.0
    cost = 20.0
    insert_charging_record(user_id, charger_id, start_time, end_time, charging_amount, cost)
    return jsonify({'success': True, 'message': '充电结束', 'end_time': end_time})

@maintenance_bp.route('/submit_repair_request/<int:charger_id>', methods=['GET', 'POST'])
def submit_repair_request(charger_id):
    if 'username' not in session:
        return redirect(url_for('maintenance.login'))
    if request.method == 'POST':
        description = request.form.get('description')
        user_id = session.get('user_id')
        request_time = datetime.now()
        try:
            conn = sqlite3.connect(r'dataset_charge_pile\datas\charging_data.db')
            cursor = conn.cursor()
            cursor.execute("INSERT INTO repair_requests (user_id, charger_id, request_time, description) VALUES (?,?,?,?)",
                           (user_id, charger_id, request_time, description))
            conn.commit()
            conn.close()
            return redirect(url_for('maintenance.repair_requests'))
        except Exception as e:
            return f"提交失败: {str(e)}"
    return render_template('submit_repair_request.html', charger_id=charger_id)

@maintenance_bp.route('/handle_repair_request/<int:request_id>', methods=['GET', 'POST'])
def handle_repair_request(request_id):
    if 'username' not in session or ('role' not in session or session['role'] != 'admin'):
        return redirect(url_for('maintenance.login'))
    if request.method == 'POST':
        admin_id = get_user_id(session['username'])
        response_text = request.form.get('response_text')
        response_time = datetime.now()
        try:
            conn = sqlite3.connect(r'dataset_charge_pile\datas\charging_data.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO repair_responses (request_id, admin_id, response_time, response_text)
                VALUES (?,?,?,?)
            ''', (request_id, admin_id, response_time, response_text))
            cursor.execute('''
                UPDATE repair_requests
                SET status = 'handled'
                WHERE id =?
            ''', (request_id,))
            conn.commit()
            conn.close()
            return jsonify({'success': True, 'message': '报修请求已处理'})
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})
    conn = sqlite3.connect(r'dataset_charge_pile\datas\charging_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT charger_id, description
        FROM repair_requests
        WHERE id =?
    ''', (request_id,))
    data = cursor.fetchone()
    conn.close()
    if data:
        charger_id, description = data
        return render_template('handle_repair_request.html', request_id=request_id, charger_id=charger_id, description=description)
    return "未找到该报修请求"

@maintenance_bp.route('/repair_requests', methods=['GET'])
def repair_requests():
    if 'username' not in session:
        return redirect(url_for('maintenance.login'))
    try:
        conn = sqlite3.connect(r'dataset_charge_pile\datas\charging_data.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM repair_requests")
        requests = cursor.fetchall()
        conn.close()
        return render_template('repair_requests.html', requests=requests)
    except Exception as e:
        return f"获取报修请求列表失败: {str(e)}"

@maintenance_bp.route('/upload_qrcode', methods=['GET', 'POST'])
def upload_qrcode():
    if 'username' not in session or ('role' not in session or session['role'] != 'admin'):
        return redirect(url_for('maintenance.login'))
    if request.method == 'POST':
        charger_id = request.form.get('charger_id')
        qrcode = request.files['qrcode']
        if qrcode:

            qrcode_path = os.path.join('static/qrcodes', f'{charger_id}_{datetime.now().strftime("%Y%m%d%H%M%S")}.png')
            qrcode.save(qrcode_path)
            try:
                conn = sqlite3.connect(r'dataset_charge_pile\datas\charging_data.db')
                cursor = conn.cursor()
                cursor.execute("INSERT INTO recharge_qrcodes (charger_id, qrcode_path, upload_time) VALUES (?,?,?)",
                               (charger_id, qrcode_path, datetime.now()))
                conn.commit()
                conn.close()
                return redirect(url_for('maintenance.charger_details'))
            except Exception as e:
                return f"上传失败: {str(e)}"
    return render_template('upload_qrcode.html')

@maintenance_bp.route('/recharge', methods=['GET', 'POST'])
def recharge():
    if 'username' not in session:
        return redirect(url_for('maintenance.login'))
    if request.method == 'POST':
        charger_id = request.form.get('charger_id')
        user_id = get_user_id(session['username'])
        try:
            conn = sqlite3.connect(r'dataset_charge_pile\datas\charging_data.db')
            cursor = conn.cursor()

            cursor.execute("INSERT INTO recharge_records (user_id, charger_id, recharge_time) VALUES (?,?,?)",
                           (user_id, charger_id, datetime.now()))
            conn.commit()
            conn.close()
            return "充值成功（模拟）"
        except Exception as e:
            return f"充值失败: {str(e)}"
    conn = sqlite3.connect(r'dataset_charge_pile\datas\charging_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT charger_id, qrcode_path FROM recharge_qrcodes")
    qrcodes = cursor.fetchall()
    conn.close()
    return render_template('recharge.html', qrcodes=qrcodes)

@maintenance_bp.route('/submit_feedback', methods=['GET', 'POST'])
def submit_feedback():
    if 'username' not in session:
        return redirect(url_for('maintenance.login'))
    if request.method == 'POST':
        feedback_type = request.form.get('feedback_type')
        content = request.form.get('content')
        user_id = get_user_id(session['username'])
        submit_time = datetime.now()
        try:
            conn = sqlite3.connect(r'dataset_charge_pile\datas\charging_data.db')
            cursor = conn.cursor()
            cursor.execute("INSERT INTO feedback (user_id, feedback_type, content, submit_time) VALUES (?,?,?,?)",
                           (user_id, feedback_type, content, submit_time))
            conn.commit()
            conn.close()
            return redirect(url_for('maintenance.feedback_list'))
        except Exception as e:
            return f"提交失败: {str(e)}"
    return render_template('submit_feedback.html')

@maintenance_bp.route('/feedback_list', methods=['GET'])
def feedback_list():
    if 'username' not in session:
        return redirect(url_for('maintenance.login'))
    try:
        conn = sqlite3.connect(r'dataset_charge_pile\datas\charging_data.db')
        cursor = conn.cursor()
        cursor.execute("SELECT id, feedback_type, content, submit_time FROM feedback")
        feedbacks = cursor.fetchall()
        conn.close()
        return render_template('feedback_list.html', feedbacks=feedbacks)
    except Exception as e:
        return f"获取反馈列表失败: {str(e)}"

@maintenance_bp.route('/charging_report')
def charging_report():
    if 'username' not in session or ('role' not in session or session['role'] != 'admin'):
        return redirect(url_for('maintenance.login'))
    charging_records = get_charging_records_for_report()
    return render_template('charging_report.html', charging_records=charging_records)

@maintenance_bp.route('/repair_report')
def repair_report():
    if 'username' not in session or ('role' not in session or session['role'] != 'admin'):
        return redirect(url_for('maintenance.login'))
    repair_requests = get_repair_requests_for_report()
    return render_template('repair_report.html', repair_requests=repair_requests)