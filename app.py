import streamlit as st
import pandas as pd
import firebase_admin
from firebase_admin import credentials, db
import joblib
import numpy as np
import json
import altair as alt
import time

st.set_page_config(page_title="AI Fault Diagnosis System", layout="wide")

# Firebase initialization (FOR STREAMLIT CLOUD)
if not firebase_admin._apps:
    cred_dict = json.loads(st.secrets["FIREBASE_CREDENTIALS"])
    cred = credentials.Certificate(cred_dict)

    firebase_admin.initialize_app(cred, {
        "databaseURL": st.secrets["DATABASE_URL"]
    })

sensor_ref = db.reference('sensor_data')

# Load ML model
model = joblib.load('random_forest_model.pkl')

st.title("AI-Powered Fault Diagnosis System Dashboard")
tabs = st.tabs(["Fault Status", "Historical Data", "Sensor Plots"])

# --- Helper functions ---
def preprocess_features(entry):
    return pd.DataFrame([{
        "temperature": entry.get("temperature", 0),
        "vibration": entry.get("vibration", 0),
        "current": entry.get("current", 0)
    }])

def fetch_firebase_data():
    try:
        data_snapshot = sensor_ref.get()
        data_list = []

        if data_snapshot:
            for key in sorted(data_snapshot.keys()):
                entry = data_snapshot[key]
                entry['Timestamp'] = entry.get('time', key)

                prediction = model.predict(preprocess_features(entry))[0]
                entry['Fault_Status'] = "Faulty" if prediction == 1 else "Normal"

                data_list.append(entry)

        return data_list

    except Exception as e:
        st.error(f"Firebase connection error: {e}")
        return []

data_list = fetch_firebase_data()

# --- Historical Data Tab ---
with tabs[1]:
    if data_list:
        df_hist = pd.DataFrame(data_list)
        st.dataframe(df_hist)
    else:
        st.info("No historical data available.")

# --- Fault Status Tab ---
with tabs[0]:
    if data_list:
        latest = data_list[-1]
        status = latest["Fault_Status"]
        st.header(f"Status: {status}")
    else:
        st.info("No data available.")

# --- Sensor Plots Tab ---
with tabs[2]:
    if data_list:
        df = pd.DataFrame(data_list)
        st.line_chart(df[["temperature", "vibration", "current"]])
    else:
        st.info("No sensor data available.")

time.sleep(5)
st.rerun()