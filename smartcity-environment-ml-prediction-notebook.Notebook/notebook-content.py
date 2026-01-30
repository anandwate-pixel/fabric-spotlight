# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "ce7647fb-b572-446d-a575-aee6b2d69c25",
# META       "default_lakehouse_name": "smartcity_environment_lakehouse_gold",
# META       "default_lakehouse_workspace_id": "b32ada8b-2770-4a2d-889e-74882bf450d1",
# META       "known_lakehouses": [
# META         {
# META           "id": "ce7647fb-b572-446d-a575-aee6b2d69c25"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import pyspark.sql.functions as F
from prophet import Prophet

# --------------------------------------------------
# 1. Time Configuration (IST)
# --------------------------------------------------
ist = pytz.timezone("Asia/Kolkata")

current_ist = datetime.now(ist).replace(minute=0, second=0, microsecond=0)

future_times = pd.date_range(
    start=current_ist + timedelta(hours=1),
    periods=6,
    freq="H",
    tz=ist
)

# --------------------------------------------------
# 2. Load Cities
# --------------------------------------------------
cities = (
    spark.read.table(
        "smartcity_environment_lakehouse_gold.smartcity_environment_weather_traffic_table"
    )
    .select("city")
    .distinct()
    .toPandas()["city"]
)

# --------------------------------------------------
# 3. Prophet Training Function
# --------------------------------------------------
def train_prophet(df, time_col, target_col):
    prophet_df = (
        df[[time_col, target_col]]
        .dropna()
        .rename(columns={time_col: "ds", target_col: "y"})
    )

    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"]).dt.tz_localize(None)
    prophet_df = prophet_df.sort_values("ds")

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False
    )
    model.fit(prophet_df)

    return model

# --------------------------------------------------
# 4. Forecast Loop Per City
# --------------------------------------------------
all_forecasts = []

for city_name in cities:
    print(f"🔵 Processing city: {city_name}")

    df_pd = (
        spark.read.table(
            "smartcity_environment_lakehouse_gold.smartcity_environment_weather_traffic_table"
        )
        .filter(F.col("city") == city_name)
        .toPandas()
    )

    if df_pd.empty:
        continue

    # Ensure datetime columns exist
    df_pd["weather_time"] = pd.to_datetime(df_pd["weather_time"]).dt.tz_localize(None)
    df_pd["traffic_time"] = pd.to_datetime(df_pd["traffic_time"]).dt.tz_localize(None)

    # -----------------------------
    # Train Prophet Models
    # -----------------------------
    temp_model = train_prophet(df_pd, "weather_time", "temp_c")
    wind_model = train_prophet(df_pd, "weather_time", "wind_kph")

    speed_model = train_prophet(df_pd, "traffic_time", "currentSpeed")
    congestion_model = train_prophet(df_pd, "traffic_time", "congestion")

    # -----------------------------
    # Future DataFrames
    # -----------------------------
    future_weather = pd.DataFrame({
        "ds": future_times.tz_localize(None)
    })

    future_traffic = pd.DataFrame({
        "ds": future_times.tz_localize(None)
    })

    # -----------------------------
    # Predictions
    # -----------------------------
    temp_fc = temp_model.predict(future_weather)
    wind_fc = wind_model.predict(future_weather)

    speed_fc = speed_model.predict(future_traffic)
    congestion_fc = congestion_model.predict(future_traffic)

    # -----------------------------
    # Final Output Per City
    # -----------------------------
    forecast_df = pd.DataFrame({
        "city": city_name,
        "localtime": future_times.astype(str),
        "Predicted_temp_c": temp_fc["yhat"].values,
        "Predicted_wind_kph": wind_fc["yhat"].values,
        "Predicted_AvgSpeed_kmph": speed_fc["yhat"].values,
        "Predicted_Congestion": np.clip(congestion_fc["yhat"].values, 0, 100)
    })

    all_forecasts.append(forecast_df)

# --------------------------------------------------
# 5. Combine All Cities
# --------------------------------------------------
final_forecast_df = pd.concat(all_forecasts, ignore_index=True)
display(final_forecast_df)

# --------------------------------------------------
# 6. Save to Lakehouse
# --------------------------------------------------

# Historical append
spark.createDataFrame(final_forecast_df) \
    .write.mode("append") \
    .format("delta") \
    .saveAsTable(
        "smartcity_environment_lakehouse_gold.smartcity_environment_weather_traffic_forecastnext6hours_ml_historical_current"
    )

# Latest snapshot overwrite
spark.createDataFrame(final_forecast_df) \
    .write.mode("overwrite") \
    .format("delta") \
    .saveAsTable(
        "smartcity_environment_lakehouse_gold.smartcity_environment_weather_traffic_forecastnext6hours_ml_forecast"
    )

print("✅ Prophet-based forecasts saved successfully")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
