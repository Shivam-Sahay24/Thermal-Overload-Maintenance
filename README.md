# Thermal-Overload-Maintenance

A real-time predictive maintenance dashboard for Autonomous Underwater Vehicle (AUV) thrusters. This project utilizes a PyTorch Gated Recurrent Unit (GRU) to forecast thermal overload events based on live subsea electrical telemetry.

## ⚙️ Engineering Overview
When an AUV ingests debris, the propeller stalls, causing a massive current spike that degrades the motor coils before the physical thermistor registers the heat. 

This Digital Twin bypasses absolute temperature thresholds by predicting the thermal trajectory **10 milliseconds** into the future based on a 50-step sliding window of hardware telemetry:
* Command PWM (μs)
* Current Draw (A)
* Voltage (V)
* Actuator Angle (°)
* Mechanical Thrust (g)

## Machine Learning Architecture
* **Competitors:** LSTM vs. GRU Showdown
* **Winner:** The GRU achieved a superior Validation MSE of `0.000044` compared to the LSTM's `0.000130`. 
* **Optimization:** AdamW optimizer with a `1e-4` weight decay was utilized to successfully prevent overfitting on the noisy hardware telemetry.

##  How to Run Locally
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/AUV_Digital_Twin.git](https://github.com/YOUR_USERNAME/AUV_Digital_Twin.git)
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
3. Launch the Streamlit Interface:
   ```bash
   python -m streamlit run app.py
