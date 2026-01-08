# Ukraine Missile & UAV Strikes Analysis: Deep Learning Forecasting (LSTM & CNN)

## Executive Summary
This project delivers a comprehensive analytical framework and predictive engine for tracking Russian missile and Unmanned Aerial Vehicle (UAV) strikes on Ukrainian infrastructure (October 2022 – Present). Moving beyond simple statistical counting, this initiative applies advanced **Deep Learning (LSTM & CNN)** to forecast future attack volumes based on historical time-series data.

By leveraging a dataset of verified military reports, I developed a pipeline that not only visualizes the geospatial evolution of the conflict but also trains neural networks to anticipate surge periods. The **Long Short-Term Memory (LSTM)** model achieved a **12% improvement in Mean Absolute Error (MAE)** over comparative CNN architectures, providing a viable proof-of-concept for automated threat forecasting and resource allocation.

## Data Source & Attribution
The core dataset is manually curated from official reports by the **Air Force Command of UA Armed Forces** and the **General Staff of the Armed Forces of Ukraine**.
* **Primary Source:** Verified reports via official Facebook/Telegram channels.
* **Dataset:** `missile_attacks_daily.csv` (Time-series launch data).
* **Key Features:** `launch_place`, `model` (Missile/UAV type), `launched` vs `destroyed` counts, and `target` coordinates.
* **Geospatial Scope:** Covers all major administrative regions (oblasts) of Ukraine.

## Technical Architecture & Stack
* **Language:** Python 3.9+
* **Deep Learning:** **TensorFlow/Keras** (LSTM, 1D-CNN implementation).
* **Data Processing:** **Pandas** & **NumPy** (Time-series windowing, normalization).
* **Visualization:** **Matplotlib** & **Seaborn** (Geospatial scatter plots, loss curves).
* **Infrastructure:** GPU-accelerated training environment for sequential tensor operations.

## Visual Analysis & Strategic Insights

### Geospatial Attack Vectors
To understand the strategic "encirclement" of air defenses, I mapped the origin points of all recorded strikes.

<p align="center">
  <img src=".assets/Frequency of Missile Strikes.png" alt="Geospatial Frequency of Strikes" width="800"/>
  <br>
  <b>Figure 1: Geographical Frequency of Missile Attacks by Region</b>
</p>

The scatter plot reveals distinct clusters corresponding to major launch sites:
* **South (Crimea):** Dense activity at Chauda (45.0°N, 35.8°E), a primary vector for Shahed drone launches.
* **Southeast (Krasnodar):** High volume from Primorsko-Akhtarsk (46.0°N, 38.2°E).
* **North (Russia):** Launches from Bryansk/Kursk oblasts (52-54°N).
**Strategic Insight:** This confirms a multi-vector attack strategy designed to saturate air defenses from the North, East, and South simultaneously.

### Regional Vulnerability Assessment
Understanding where impacts occur helps identify high-risk zones.

<p align="center">
  <img src=".assets/Top 15 Affected Regions.png" alt="Top 15 Affected Regions" width="800"/>
  <br>
  <b>Figure 2: Top 15 Affected Regions</b>
</p>

**Kyiv** dominates as the primary target due to its strategic significance, followed by logistics hubs like **Dnipropetrovsk** and **Odesa**. This consistent targeting pattern confirms that "Region" is a high-weight feature for predictive modeling.

## Predictive Modeling: LSTM vs. CNN

To forecast the number of launched units ($t+1$), I developed and compared two deep learning architectures.

### 1. Long Short-Term Memory (LSTM)
* **Architecture:** Two stacked LSTM layers (to capture long-term temporal dependencies) followed by a Dense output layer.
* **Training:** Sliding window approach (30-day lookback), Adam optimizer (lr=0.001), Dropout (0.2) to prevent overfitting.
* **Rationale:** Chosen for its ability to mitigate the vanishing gradient problem and learn "strategic pauses" in the conflict timeline.

### 2. 1D Convolutional Neural Network (CNN)
* **Architecture:** 1D Convolutional layer (Kernel size=7 to capture weekly patterns) -> MaxPooling -> Dense layer.
* **Rationale:** Treats the time-series as a 1D signal to extract short-term local patterns (e.g., week-long campaigns).

## Evaluation & Results

### Forecasting Capabilities
The models were evaluated on their ability to predict daily launch counts on unseen test data.

<p align="center">
  <img src=".assets/Missile Attacks Forecasting.png" alt="LSTM Forecasting Results" width="800"/>
  <br>
  <b>Figure 3: Missile Attacks Forecasting (LSTM)</b>
</p>

The LSTM (Orange line) successfully captures the "cadence" of the conflict, anticipating the timing of major spikes. While it tends to be conservative regarding the *magnitude* of extreme outliers (300+ launches), it correctly identifies the *onset* of high-activity periods.

### Model Comparison
I compared the two architectures directly to determine the optimal solution for deployment.

<p align="center">
  <img src=".assets/LSTM vs CNN on Test Data.png" alt="LSTM vs CNN Prediction" width="800"/>
  <br>
  <b>Figure 4: LSTM vs CNN Prediction on Test Data</b>
</p>

**Visual Analysis:** The LSTM (Orange) produces a smoother, more stable trend line that accounts for longer-term history. The CNN (Green) is more reactive to short-term noise, resulting in sharper fluctuations but less overall accuracy.

### Quantitative Performance Metrics

| Model Architecture | MAE (Lower is Better) | RMSE | Training Time |
| :--- | :--- | :--- | :--- |
| **LSTM (Proposed)** | **12.4** | **18.2** | 50 Epochs |
| 1D-CNN | 14.1 | 22.5 | 35 Epochs |
| Baseline (Linear Reg) | 28.5 | 34.0 | N/A |

**Conclusion:** The LSTM achieved a **~12% improvement** in Mean Absolute Error (MAE) compared to the CNN, validating its superiority for this specific time-series application where past context significantly influences future events.

## Conclusion & Future Work
This project demonstrates that aerial warfare, while seemingly chaotic, follows detectable patterns rooted in logistics and strategy. By transitioning from retrospective analysis to **predictive deep learning**, we can provide actionable intelligence on future threat levels.

**Future Recommendations:**
1.  **Multivariate Integration:** Incorporate external datasets (e.g., Black Sea warship movements) to provide leading indicators for sea-based launches (Kalibr missiles).
2.  **Real-Time Pipeline:** Automate data scraping from Telegram sources to transform this from a static analysis into a live early-warning system.

## References
* **BBC News:** *Russian attacks on Ukraine double since Trump inauguration*
* **CSIS:** *Calculating the cost-effectiveness of Russia’s drone strikes*
* **War Quants:** *Sustained Russian Shahed swarms: The war of precision mass continues*
* **Pavlo Krasnomovets:** *Missile Attacks Calendar*
