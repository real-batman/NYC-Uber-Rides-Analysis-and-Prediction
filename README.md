# NYC-Uber-Rides-Analysis-and-Prediction

An end-to-end data analysis project on 564K Uber pickup records from New York City (April 2014), covering exploratory data analysis, feature engineering, and predictive modelling for hourly ride demand.

---

## Project Overview

This project analyzes Uber trip data to uncover demand patterns across time and geography, then builds a machine learning model to forecast hourly ride demand. The goal is to simulate a real-world consulting/analytics engagement — turning raw trip logs into operational recommendations a ride-hailing company could act on.

**Key questions answered:**
- When do demand peaks occur, and how much do they vary by day/hour?
- Which dispatch bases handle the most supply, and is there concentration risk?
- Can hourly trip volume be predicted with high accuracy using historical patterns?

---


## Tech Stack

- **Language:** Python 3.10
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn (Linear Regression, Ridge, Random Forest, Gradient Boosting)
- **Environment:** Jupyter Notebook


## Methodology

### 1. Data Cleaning
- Parsed raw datetime strings into proper `datetime` objects
- Removed duplicates and null rows
- Applied a geographic bounding-box filter to drop invalid NYC coordinates

### 2. Feature Engineering
Derived 7+ temporal features from the raw timestamp:
- `hour`, `day_of_week`, `day_name`, `week`
- `is_weekend` — weekend flag
- `is_rush_hour` — weekday 7–9 AM / 5–7 PM flag
- `time_of_day` — Night / Morning / Afternoon / Evening buckets
- `lag_1h`, `lag_24h`, `rolling_3h_mean` — time-series lag features for forecasting

### 3. Exploratory Data Analysis (8 visualizations)
- Trips by hour of day
- Trips by day of week
- Daily trend across the month
- Hour × day-of-week heatmap
- Trips by dispatch base (bar + pie)
- Geographic scatter plot of pickups
- Rush-hour vs non-rush-hour comparison
- Time-of-day distribution

### 4. Predictive Modelling
Aggregated trip-level data into an **hourly demand time series**, then trained and compared 4 regression models using a **chronological 80/20 train-test split** (no data leakage):

| Model | R² | MAE (trips/hr) |
|---|---|---|
| Linear Regression | 0.71 | 142 |
| Ridge Regression | 0.72 | 138 |
| Random Forest | 0.91 | 58 |
| **Gradient Boosting (Champion)** | **0.95** | **31** |

---

## Key Findings

| Finding | Detail |
|---|---|
| **Peak demand window** | 17:00–19:00 (evening rush) |
| **Busiest day** | Friday — 24% higher than Monday |
| **Supply concentration risk** | Base `B02617` handles ~50% of all trips |
| **Best predictor** | `lag_1h` (prior hour's demand) — 38% feature importance |
| **Model performance** | Gradient Boosting achieves R² = 0.95, MAE = 31 trips/hour |

---

## 💡 Recommendations

1. **Pre-position drivers by 16:30** ahead of the evening demand surge
2. **Apply dynamic pricing** on Friday evenings to balance supply and demand
3. **Diversify dispatch base capacity** — reduce reliance on B02617 to mitigate single-point-of-failure risk
4. **Deploy the Gradient Boosting model** for real-time, hour-ahead operational dispatch planning

