import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time

st.set_page_config(
    page_title="Thermal Overload Forecasting",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {background-color: #0E1117;}
    h1 {color: #00E6FF; font-family: 'Courier New', Courier, monospace;}
    .stMetric {background-color: #1E2127; padding: 15px; border-radius: 10px; border-left: 5px solid #00E6FF;}
    </style>
    """, unsafe_allow_html=True)

class ThermalGRU(nn.Module):
    def __init__(self, input_size=6, hidden_size=32):
        super(ThermalGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        final_time_step = out[:, -1, :]
        return self.fc(final_time_step)

@st.cache_resource
def load_assets():
    model = ThermalGRU(input_size=6, hidden_size=32)
    model.load_state_dict(torch.load("thermal_gru_weights.pth"))
    model.eval()
    scaler = joblib.load("thermal_scaler.pkl")
    return model, scaler

model, scaler = load_assets()

st.sidebar.markdown("""
    <div style="text-align: center; padding: 20px 10px; background-color: #1E2127; border-radius: 10px; border-bottom: 4px solid #00E6FF; margin-bottom: 20px;">
        <h2 style="color: white; margin: 0; font-family: 'Courier New', Courier, monospace;">
            ThrusterData <span style="color: #00E6FF;"></span>
        </h2>
        <p style="color: #888888; font-size: 12px; margin-top: 5px; letter-spacing: 1px;">
            BASED ON RAW TELEMETRY
        </p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("Override Controls")

pwm_input = st.sidebar.slider("Command PWM (μs)", min_value=900, max_value=2000, value=1500, step=10)

fig_dial = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = pwm_input,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "PWM Throttle", 'font': {'color': '#00E6FF', 'size': 14}},
    gauge = {
        'axis': {'range': [900, 2000], 'tickwidth': 1, 'tickcolor': "white"},
        'bar': {'color': "#00E6FF"},
        'bgcolor': "#1E2127",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [900, 1400], 'color': '#0E1117'},
            {'range': [1400, 1800], 'color': '#1a2639'},
            {'range': [1800, 2000], 'color': '#4a1515'} # Redline zone
        ],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': 1850 # Visual redline for the thruster
        }
    }
))

fig_dial.update_layout(
    height=250, 
    margin=dict(l=10, r=10, t=30, b=10),
    paper_bgcolor='#0E1117',
    font={'color': "white"}
)

st.sidebar.plotly_chart(fig_dial, use_container_width=True)

st.sidebar.markdown("### Secondary Telemetry")
col_a, col_b = st.sidebar.columns(2)
with col_a:
    thrust_input = st.slider("Thrust (g)", value=150.0, max_value= 2500.0, step=5.0)
    current_input = st.slider("Current (A)", value=2.5, max_value = 25.0, step=0.1)
with col_b:
    voltage_input = st.slider("Voltage (V)", value=12.8, max_value = 14.0, step=0.1)
    angle_input = st.slider("Angle (°)", value=90.0, max_value=180.0,step=1.0)

current_temp = st.slider("Current Temp (°C)", value=45.0, max_value =100.0, step=0.5)

simulated_history = np.array([[angle_input, pwm_input, voltage_input, current_input, thrust_input, current_temp]] * 50)

scaled_history = scaler.transform(simulated_history)


tensor_input = torch.FloatTensor(scaled_history).unsqueeze(0)


with torch.no_grad():
    scaled_prediction = model(tensor_input).numpy()


dummy_array = np.zeros((1, 6))
dummy_array[0, 5] = scaled_prediction[0, 0]
predicted_temp = scaler.inverse_transform(dummy_array)[0, 5]


st.title("AUV Thruster Thermal Maintenance")
st.markdown("Real-time inference using a Gated Recurrent Unit (GRU) to forecast thermal overload events based on subsea telemetry.")

power_watts = voltage_input * current_input
efficiency = abs(thrust_input) / power_watts if power_watts > 0 else 0

is_critical = False
is_elevated = False
fault_reason = ""

if (pwm_input > 1700 or pwm_input < 1300) and (current_input > 12.0) and (abs(thrust_input) < 50.0):
    is_critical = True
    fault_reason = "MECHANICAL STALL DETECTED"

elif predicted_temp >= 75.0:
    is_critical = True
    fault_reason = "THERMAL LIMIT EXCEEDED"

elif predicted_temp >= 60.0:
    is_elevated = True
    fault_reason = "OPERATING TEMPERATURE HIGH"
    
elif predicted_temp >= 50.0 and current_input >= 10.0:
    is_elevated = True
    fault_reason = "HIGH CURRENT DRAW DEGRADING THERMAL OVERHEAD"

col1, col2, col3, col4 = st.columns(4)
temp_delta = predicted_temp - current_temp

with col1:
    st.metric(label="Current Temp", value=f"{current_temp:.1f} °C")
with col2:
    delta_color = "normal" if temp_delta < 2.0 else "inverse" 
    st.metric(label="Forecasted Temp (+10ms)", value=f"{predicted_temp:.2f} °C", delta=f"{temp_delta:.2f} °C", delta_color=delta_color)
with col3:
    st.metric(label="Power Draw", value=f"{power_watts:.1f} W", help="Voltage × Current")
    
with col4:
    if is_critical:
        st.error("CRITICAL FAULT")
        st.caption(f"**CAUSE:** {fault_reason}")
    elif is_elevated:
        st.warning("SYSTEM STRESSED")
        st.caption(f"**CAUSE:** {fault_reason}")
    else:
        st.success("SYSTEM NOMINAL")
        st.caption("All parameters within safe limits.")

st.markdown("---")

st.markdown("# Thermal Trajectory")

time_x = list(range(-50, 1))
temp_y = [current_temp - (temp_delta * (i/50)) for i in range(50)] + [predicted_temp]

fig = go.Figure()


fig.add_trace(go.Scatter(x=time_x[:-1], y=temp_y[:-1], mode='lines', name='Historical Trajectory', line=dict(color='#00E6FF', width=3)))

fig.add_trace(go.Scatter(x=[0], y=[predicted_temp], mode='markers', name='GRU Forecast', marker=dict(color='red', size=12, symbol='star')))

fig.update_layout(
    plot_bgcolor='#0E1117',
    paper_bgcolor='#0E1117',
    font_color='white',
    xaxis_title="Time Steps (ms)",
    yaxis_title="Temperature (°C)",
    margin=dict(l=0, r=0, t=30, b=0),
    showlegend=True
)

st.plotly_chart(fig, use_container_width=True)