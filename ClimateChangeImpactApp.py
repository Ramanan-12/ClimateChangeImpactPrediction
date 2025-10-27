import streamlit as st
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import platform, os, time, plotly.express as px
from PIL import Image

st.set_page_config(page_title="🌦️ AI WeatherSense+ Pro", layout="wide")

# --------------------- Model ---------------------
class LSTMWeather(nn.Module):
    def __init__(self, input_size=3, hidden_size=50, num_layers=1, output_size=3):
        super(LSTMWeather, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# --------------------- Utility ---------------------
def send_notification(title, message):
    try:
        if platform.system().lower() == "darwin":
            os.system(f'''osascript -e 'display notification "{message}" with title "{title}"' ''')
        elif platform.system().lower() == "windows":
            from plyer import notification
            notification.notify(title=title, message=message, timeout=5)
        else:
            os.system(f'notify-send "{title}" "{message}"')
    except:
        pass

@st.cache_data(ttl=3600)
def get_location():
    try:
        res = requests.get("https://ipapi.co/json").json()
        return res["latitude"], res["longitude"], res["city"]
    except:
        return 13.0827, 80.2707, "Chennai"

def fetch_weather(lat, lon, days=5):
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days)
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,relative_humidity_2m,precipitation"
    res = requests.get(url).json()
    df = pd.DataFrame({
        "time": res["hourly"]["time"],
        "temp": res["hourly"]["temperature_2m"],
        "humidity": res["hourly"]["relative_humidity_2m"],
        "rain": res["hourly"]["precipitation"]
    })
    df["time"] = pd.to_datetime(df["time"])
    return df

def predict_weather(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[["temp", "humidity", "rain"]])
    seq_len = 24
    X, y = [], []
    for i in range(len(scaled) - seq_len):
        X.append(scaled[i:i+seq_len])
        y.append(scaled[i+seq_len])
    X = np.array(X)
    y = np.array(y)
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    model = LSTMWeather()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(8):
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.no_grad():
        pred = model(torch.FloatTensor(scaled[-seq_len:].reshape(1, seq_len, 3))).numpy()
    return scaler.inverse_transform(pred)[0]

def generate_tips(temp, rain, humidity):
    tips = []
    if rain > 2:
        tips.append("☔ Rain expected soon — carry an umbrella.")
    if temp > 35:
        tips.append("🔥 Very hot — stay hydrated and avoid sunlight.")
    if temp < 18:
        tips.append("🧥 Chilly weather — wear warm clothes.")
    if humidity > 85:
        tips.append("💧 High humidity — take precautions indoors.")
    if not tips:
        tips.append("🌤️ Pleasant weather ahead. Enjoy your day!")
    return tips

# --------------------- Sidebar ---------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1116/1116453.png", width=80)
st.sidebar.title("AI WeatherSense+ Pro")
menu = st.sidebar.radio("🔍 Navigation", ["🏠 Home", "📊 Climate Trends", "🔮 AI Forecast", "⚙️ Settings"])

lat, lon, city = get_location()
df = fetch_weather(lat, lon)
pred_temp, pred_hum, pred_rain = predict_weather(df)
tips = generate_tips(pred_temp, pred_rain, pred_hum)

# --------------------- Home Page ---------------------
if menu == "🏠 Home":
    st.markdown("<h2 style='text-align:center; color:#4A90E2;'>🌈 Smart Recommendations</h2>", unsafe_allow_html=True)
    st.info(f"📍 Location: **{city}** ({lat:.2f}, {lon:.2f})")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🌡️ Temperature", f"{pred_temp:.2f}°C")
    with col2:
        st.metric("💧 Humidity", f"{pred_hum:.2f}%")
    with col3:
        st.metric("☔ Rainfall", f"{pred_rain:.2f} mm")

    st.subheader("💬 Recommendations")
    for tip in tips:
        st.success(tip)
        send_notification("🌦️ Weather Alert", tip)

    st.toast("✅ Recommendations updated!", icon="🌦️")

# --------------------- Climate Trends ---------------------
elif menu == "📊 Climate Trends":
    st.header("📊 Climate Trends - Past 5 Days")
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.line(df, x="time", y="temp", title="🌡️ Temperature Trend", color_discrete_sequence=["red"])
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.line(df, x="time", y="humidity", title="💧 Humidity Trend", color_discrete_sequence=["blue"])
        st.plotly_chart(fig2, use_container_width=True)
    st.plotly_chart(px.bar(df, x="time", y="rain", title="☔ Rainfall Trend", color_discrete_sequence=["green"]), use_container_width=True)

# --------------------- Forecast Page ---------------------
elif menu == "🔮 AI Forecast":
    st.header("🔮 AI Predicted Weather (Next 24 Hours)")
    st.metric("🌡️ Temperature", f"{pred_temp:.2f}°C")
    st.metric("💧 Humidity", f"{pred_hum:.2f}%")
    st.metric("☔ Rainfall", f"{pred_rain:.2f} mm")
    st.warning("⚙️ Model trained with past 5-day data using LSTM (AI-powered).")

# --------------------- Settings ---------------------
elif menu == "⚙️ Settings":
    st.header("⚙️ Application Settings")
    refresh = st.slider("🔁 Auto-refresh interval (minutes)", 5, 60, 15)
    notify = st.checkbox("🔔 Enable Notifications", True)
    st.info(f"✅ Notifications {'enabled' if notify else 'disabled'}. Refresh interval: {refresh} minutes.")
