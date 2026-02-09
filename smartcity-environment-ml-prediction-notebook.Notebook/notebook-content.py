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
import pytz
from datetime import datetime, timedelta
import pyspark.sql.functions as F
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

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
# 3. Forecast Loop Per City
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

    # Convert to numeric
    for col in ["temp_c", "wind_kph", "currentSpeed", "congestion"]:
        df_pd[col] = pd.to_numeric(df_pd[col], errors="coerce")

    # Drop NaNs
    df_pd = df_pd.dropna(subset=["temp_c", "wind_kph", "currentSpeed", "congestion"])

    # Feature engineering
    df_pd["weather_time"] = pd.to_datetime(df_pd["weather_time"]).dt.tz_localize(None)
    df_pd["hour"] = df_pd["weather_time"].dt.hour
    df_pd["dayofweek"] = df_pd["weather_time"].dt.dayofweek

    # -----------------------------
    # Train regression model for temp, speed, congestion
    # -----------------------------
    X = df_pd[["hour", "dayofweek"]]
    y = df_pd[["temp_c", "currentSpeed", "congestion"]]

    model = MultiOutputRegressor(XGBRegressor(n_estimators=200, max_depth=5))
    model.fit(X, y)

    # Future features
    future_df = pd.DataFrame({
        "hour": future_times.hour,
        "dayofweek": future_times.dayofweek
    })

    preds = model.predict(future_df)

    preds_df = pd.DataFrame(preds, columns=[
        "Predicted_temp_c", "Predicted_AvgSpeed_kmph", "Predicted_Congestion"
    ])

    # -----------------------------
    # Wind forecast: historical average + gusts
    # -----------------------------
    hourly_avg_wind = df_pd.groupby("hour")["wind_kph"].mean()
    predicted_wind = []
    for h in future_times.hour:
        base = hourly_avg_wind.get(h, df_pd["wind_kph"].mean())
        gust = np.random.normal(0, 2)   # ±2 km/h variability
        predicted_wind.append(np.clip(base + gust, 0, 150))

    preds_df["Predicted_wind_kph"] = predicted_wind

    # -----------------------------
    # Clip predictions to realistic ranges
    # -----------------------------
    preds_df["Predicted_temp_c"] = np.clip(preds_df["Predicted_temp_c"], -10, 50)
    preds_df["Predicted_AvgSpeed_kmph"] = np.clip(preds_df["Predicted_AvgSpeed_kmph"], 0, 120)
    preds_df["Predicted_Congestion"] = np.clip(preds_df["Predicted_Congestion"], 0, 100)

    forecast_df = pd.DataFrame({
        "city": city_name,
        "localtime": future_times.astype(str)
    }).join(preds_df)

    all_forecasts.append(forecast_df)

# --------------------------------------------------
# 4. Combine All Cities
# --------------------------------------------------
final_forecast_df = pd.concat(all_forecasts, ignore_index=True)
final_forecast_df = final_forecast_df.loc[:, ~final_forecast_df.columns.duplicated()]

display(final_forecast_df)

# ---------------# Remove duplicate columns
#final_forecast_df = final_forecast_df.loc[:, ~final_forecast_df.columns.duplicated()]

# 5. Save to Lakehouse
# --------------------------------------------------
spark.createDataFrame(final_forecast_df) \
    .write.mode("append") \
    .format("delta") \
    .option("mergeSchema", "false") \
    .saveAsTable(
        "smartcity_environment_lakehouse_gold.smartcity_environment_weather_traffic_forecastnext6hours_ml_historical_current"
    )

spark.createDataFrame(final_forecast_df) \
    .write.mode("overwrite") \
    .format("delta") \
    .option("mergeSchema", "false") \
    .saveAsTable(
        "smartcity_environment_lakehouse_gold.smartcity_environment_weather_traffic_forecastnext6hours_ml_forecast"
    )

print("✅ Hybrid forecasts saved successfully with realistic bounds")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
