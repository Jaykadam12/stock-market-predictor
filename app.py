from flask import Flask, render_template, request, send_file, redirect, url_for
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import logging
from flask_sqlalchemy import SQLAlchemy
from apscheduler.schedulers.background import BackgroundScheduler
from flask_mail import Mail, Message
from dotenv import load_dotenv
import secrets


load_dotenv()
# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///alerts.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USER')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASS')

print("MAIL USER:", os.getenv('MAIL_USER'))
print("MAIL PASS:", os.getenv('MAIL_PASS'))


# Initialize extensions
db = SQLAlchemy(app)
mail = Mail(app)
scheduler = BackgroundScheduler(daemon=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "models/"
os.makedirs(MODEL_PATH, exist_ok=True)

# function for testing mail is sending or not
# with app.app_context():
#     from flask_mail import Message
#     msg = Message("Test Email", sender=os.getenv('MAIL_USER'), recipients=["jaykadam559@gmail.com"])
#     msg.body = "This is a test email from your Flask app."
#     mail.send(msg)
#     print("✅ Test email sent successfully!")


# Alert Model
class Alert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), nullable=False)
    stock_symbol = db.Column(db.String(10), nullable=False)
    target_price = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)
    active = db.Column(db.Boolean, default=True)

    def __repr__(self):
        return f'<Alert {self.stock_symbol} {self.target_price}>'

# Create database tables
with app.app_context():
    db.create_all()

# Scheduler for checking prices
def check_price_alerts():
    with app.app_context():
        active_alerts = Alert.query.filter_by(active=True).all()
        for alert in active_alerts:
            try:
                stock = yf.Ticker(alert.stock_symbol)
                current_price = stock.history(period='1d')['Close'].iloc[-1]
                
                if current_price >= alert.target_price:
                    send_alert_email(alert, current_price)
                    alert.active = False
                    db.session.commit()
            except Exception as e:
                logger.error(f"Error checking alert {alert.id}: {e}")

def send_alert_email(alert, current_price):
    try:
        msg = Message("🚨 Stock Price Alert!",
                      sender=app.config['MAIL_USERNAME'],
                      recipients=[alert.email])
        msg.body = f"""
        {alert.stock_symbol} has reached your target price!
        
        Target Price: ${alert.target_price:.2f}
        Current Price: ${current_price:.2f}
        
        Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}
        """
        mail.send(msg)
    except Exception as e:
        logger.error(f"Error sending email: {e}")

scheduler.add_job(check_price_alerts, 'interval', minutes=15)
scheduler.start()

# Stock prediction functions (keep your existing code)
def fetch_stock_data(stock_name, period="2y", interval="1d"):
    try:
        stock = yf.Ticker(stock_name)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            return None
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df['price_change'] = df['Close'].pct_change() * 100  
        df.dropna(inplace=True)

        df['ma_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
        df['ma_200'] = df['Close'].rolling(window=200, min_periods=1).mean()

        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        return df
    except Exception as e:
        logger.error(f"Error fetching data: {e}", exc_info=True)
        return None

def train_and_save_model(df, stock_name):
    X = df[["year", "month", "ma_50", "ma_200"]].values  
    y = df["price_change"].values  

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    joblib.dump((model, scaler), f"{MODEL_PATH}{stock_name}.pkl")
    return model, scaler

def load_model(stock_name):
    try:
        model_path = f"{MODEL_PATH}{stock_name}.pkl"
        if os.path.exists(model_path):
            return joblib.load(model_path)
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
    return None, None

def predict_future(model, scaler, df):
    next_month = (datetime.today() + timedelta(days=30)).strftime("%B %Y")
    last_ma_50 = df['ma_50'].iloc[-1]
    last_ma_200 = df['ma_200'].iloc[-1]

    future_date = np.array([[datetime.today().year, datetime.today().month + 1, last_ma_50, last_ma_200]])
    future_date_scaled = scaler.transform(future_date)

    prediction = model.predict(future_date_scaled)[0]
    prediction = round(prediction, 2)

    return {"month": next_month, "prediction": prediction}

def generate_chart(df, stock_name):
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Close'], label='Closing Price', color='blue', linewidth=2)
    plt.plot(df['Date'], df['ma_50'], label='50-Day MA', color='orange', linestyle='dashed')
    plt.plot(df['Date'], df['ma_200'], label='200-Day MA', color='red', linestyle='dashed')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'Stock Price Chart for {stock_name}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return f'<img src="data:image/png;base64,{chart_url}" alt="Stock Chart">'

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        stock_name = request.form["stock_name"].strip().upper()
        df = fetch_stock_data(stock_name)

        if df is None or df.empty:
            return render_template("predict.html", 
                                 prediction_text="⚠️ Invalid stock symbol or no data available.",
                                 stock_symbol=stock_name)

        model, scaler = load_model(stock_name)
        if model is None:
            model, scaler = train_and_save_model(df, stock_name)

        prediction_data = predict_future(model, scaler, df)
        chart_html = generate_chart(df, stock_name)
        latest_price = df['Close'].iloc[-1]

        return render_template(
            "predict.html",
            prediction_text=f"Analysis for {stock_name}",
            predictions=prediction_data,
            chart_html=chart_html,
            latest_price=f"{latest_price:.2f}",
            stock_symbol=stock_name
        )
    return render_template("predict.html")

@app.route("/set_alert", methods=["POST"])
def set_alert():
    stock_symbol = request.form.get("stock_symbol")
    email = request.form.get("email")
    target_price = request.form.get("target_price")

    try:
        new_alert = Alert(
            email=email,
            stock_symbol=stock_symbol,
            target_price=float(target_price)
        )
        db.session.add(new_alert)
        db.session.commit()
        message = "✅ Alert set successfully! You'll receive an email when the price is reached."
    except Exception as e:
        message = f"⚠️ Error setting alert: {str(e)}"

    return render_template("predict.html",
                         alert_message=message,
                         stock_symbol=stock_symbol)

@app.route("/alerts")
def view_alerts():
    alerts = Alert.query.filter_by(active=True).all()
    return render_template("alerts.html", alerts=alerts)

@app.route("/delete_alert/<int:alert_id>")
def delete_alert(alert_id):
    alert = Alert.query.get_or_404(alert_id)
    db.session.delete(alert)
    db.session.commit()
    return redirect(url_for('view_alerts'))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Get Railway's assigned port
    app.run(host="0.0.0.0", port=port, debug=True)