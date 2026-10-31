import numpy as np
import pandas as pd

from src.constants import (GLUCOSE_COST,GLUTAMINE_COST,VCD_COST,MAB_VALUE,DOWNTIME_DAYS,PREDICTION_INTERVAL,N_SIMULATIONS,RANDOM_STATE,)

# Econ calculations
def calculate_culture_cost(glucose, glutamine, vcd):
    return (GLUCOSE_COST * glucose + GLUTAMINE_COST * glutamine + VCD_COST * vcd)

def calculate_culture_value(mab):
    return MAB_VALUE * mab

def calculate_profit(culture_value, culture_cost):
    return culture_value - culture_cost

def calculate_space_time_profit(profit,harvest_time_days,downtime_days=DOWNTIME_DAYS,):
    total_time = harvest_time_days + downtime_days

    if np.any(np.asarray(total_time) <= 0):
        raise ValueError("Total process time must be positive.")

    return profit / total_time

def calculate_space_time_yield(mab,harvest_time_days,downtime_days=DOWNTIME_DAYS,):
    total_time = harvest_time_days + downtime_days

    if np.any(np.asarray(total_time) <= 0):
        raise ValueError("Total process time must be positive.")

    return mab / total_time


def hours_to_days(time_hours):
    return time_hours / 24

# Prediction intervals
def calculate_prediction_interval(simulated_predictions,prediction_interval=PREDICTION_INTERVAL,):
    simulated_predictions = np.asarray(simulated_predictions)

    if simulated_predictions.shape[0] < 1:
        raise ValueError("At least one simulation is required.")

    if not 0 < prediction_interval < 1:
        raise ValueError("Prediction interval must be between 0 and 1.")

    alpha = 1 - prediction_interval

    lower = np.quantile(simulated_predictions,alpha / 2,axis=0,)

    median = np.quantile(simulated_predictions,0.5,axis=0,)

    upper = np.quantile(simulated_predictions,1 - alpha / 2,axis=0,)

    return lower, median, upper

def simulate_prediction_interval(model,X_new,residuals,n_simulations=N_SIMULATIONS,prediction_interval=PREDICTION_INTERVAL,random_state=RANDOM_STATE,):
    residuals = np.asarray(residuals, dtype=float).ravel()

    if len(residuals) == 0:
        raise ValueError("Residuals are required for simulation.")

    if not np.isfinite(residuals).all():
        raise ValueError("Residuals contain non-finite values.")

    if n_simulations < 1:
        raise ValueError("At least one simulation is required.")

    if not 0 < prediction_interval < 1:
        raise ValueError("Prediction interval must be between 0 and 1.")

    baseline_predictions = np.asarray(model.predict(X_new)).ravel()

    rng = np.random.default_rng(random_state)

    simulated_predictions = np.empty((n_simulations, len(baseline_predictions)))

    for simulation_id in range(n_simulations):
        sampled_residuals = rng.choice(residuals,size=len(baseline_predictions),replace=True,)

        simulated_predictions[simulation_id] = (baseline_predictions + sampled_residuals)

    lower, median, upper = calculate_prediction_interval(
        simulated_predictions,
        prediction_interval=prediction_interval,)

    return lower, median, upper, simulated_predictions


def evaluate_batch_harvest_times(batch_data,initial_glucose,initial_glutamine,initial_vcd,):
# calc for one batch
    required_columns = ["Time [h]", "mAb [mg/L]"]

    for column in required_columns:
        if column not in batch_data.columns:
            raise ValueError(f"Missing required batch column: {column}")

    if batch_data.empty:
        raise ValueError("Batch data is empty.")

    data = batch_data.copy()

    data["Time [h]"] = pd.to_numeric(data["Time [h]"],errors="raise",)
    data["mAb [mg/L]"] = pd.to_numeric(data["mAb [mg/L]"],errors="raise",)

    if data[required_columns].isna().any().any():
        raise ValueError("Time or mAb data contain missing values.")

    if (data["Time [h]"] < 0).any():
        raise ValueError("Harvest time cannot be negative.")

    if (data["mAb [mg/L]"] < 0).any():
        raise ValueError("mAb concentration cannot be negative.")

    data = data.sort_values("Time [h]").reset_index(drop=True)

    culture_cost = calculate_culture_cost(initial_glucose,initial_glutamine,initial_vcd,)

    data["Harvest Time [d]"] = hours_to_days(data["Time [h]"])
    data["Culture Cost [$/L]"] = culture_cost
    data["Culture Value [$/L]"] = calculate_culture_value(data["mAb [mg/L]"])
    data["Profit [$/L]"] = calculate_profit(data["Culture Value [$/L]"],data["Culture Cost [$/L]"],)
    data["Space-Time Profit [$/L/d]"] = calculate_space_time_profit(data["Profit [$/L]"],data["Harvest Time [d]"],)
    data["Space-Time Yield [mg/L/d]"] = calculate_space_time_yield(data["mAb [mg/L]"],data["Harvest Time [d]"],)

    return data
