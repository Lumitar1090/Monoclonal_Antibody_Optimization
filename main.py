from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.classes import BatchResultsAnalyzer
from src.constants import (PREDICTION_INTERVAL,N_SIMULATIONS,RANDOM_STATE,)
from src.functions import (evaluate_batch_harvest_times)


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "datasets"
FIGURES_DIR = PROJECT_DIR / "figures"
TABLES_DIR = PROJECT_DIR / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

analyzer = BatchResultsAnalyzer(TABLES_DIR)

initial_conditions_path = DATA_DIR / "initial_conditions.csv"

if not initial_conditions_path.exists():
    raise FileNotFoundError(f"Could not find initial_conditions.csv in {DATA_DIR}")

initial_conditions = pd.read_csv(initial_conditions_path)

required_initial_columns = ["Batch ID","Glucose [mM]","Glutamine [mM]","VCD [cells/mL]",]
missing_columns = [column
    for column in required_initial_columns
    if column not in initial_conditions.columns]

if missing_columns:
    raise ValueError(f"Missing columns in initial_conditions.csv: {missing_columns}")

initial_conditions["Batch ID"] = (pd.to_numeric(initial_conditions["Batch ID"], errors="raise").astype(int).astype(str))
batch_files = sorted(DATA_DIR.glob("batch_*.csv"))

if len(batch_files) != 30:
    raise ValueError(f"Expected 30 batch files, but found {len(batch_files)}.")

if initial_conditions["Batch ID"].duplicated().any():
    raise ValueError("Duplicate Batch ID values found in initial_conditions.csv.")


initial_ids = set(initial_conditions["Batch ID"])
file_ids = {str(int(path.stem.replace("batch_", ""))) for path in batch_files}

missing_condition_ids = file_ids - initial_ids

if missing_condition_ids:
    raise ValueError("Initial conditions are missing for these batch files: "f"{sorted(missing_condition_ids, key=int)}")

# Keep only the initial-condition rows for the selected batch files.
initial_conditions = initial_conditions[initial_conditions["Batch ID"].isin(file_ids)].copy()

# Confirm that each selected batch has exactly one conditions row.
if initial_conditions["Batch ID"].duplicated().any():
    raise ValueError(
        "Duplicate Batch ID values found in initial_conditions.csv.")

if set(initial_conditions["Batch ID"]) != file_ids:
    raise ValueError("The selected batch files and filtered initial conditions " "still do not match.")

print(f"Loaded {len(batch_files)} batch files.")
print(f"Loaded {len(initial_conditions)} initial-condition records.")
print("Number of batch files:", len(file_ids))
print("Number of initial-condition rows:", len(initial_ids))
print("Batch file IDs:", sorted(file_ids, key=int))
print("Matching IDs:", sorted(file_ids & initial_ids, key=int))

initial_conditions_by_id = initial_conditions.set_index("Batch ID")
optimization_rows = []

for batch_file in batch_files:
    batch_id = str(int(batch_file.stem.replace("batch_", "")))

    batch_data = pd.read_csv(batch_file)
    conditions = initial_conditions_by_id.loc[batch_id]

    glucose = float(conditions["Glucose [mM]"])
    glutamine = float(conditions["Glutamine [mM]"])
    vcd = float(conditions["VCD [cells/mL]"])

    harvest_results = evaluate_batch_harvest_times(
        batch_data=batch_data,
        initial_glucose=glucose,
        initial_glutamine=glutamine,
        initial_vcd=vcd,)

    harvest_results["Batch ID"] = batch_id
    harvest_results["Glucose [mM]"] = glucose
    harvest_results["Glutamine [mM]"] = glutamine
    harvest_results["VCD [cells/mL]"] = vcd


    # Select the harvest time with the highest observed space-time profit.
    best_row = harvest_results.loc[harvest_results["Space-Time Profit [$/L/d]"].idxmax()].copy()
    optimization_rows.append(best_row)

optimization_results = pd.DataFrame(optimization_rows).reset_index(drop=True)
optimization_results.to_csv(TABLES_DIR / "All_Batch_Optimization_Results.csv",index=False,)


print("\nBest observed result for each batch:")
print(optimization_results[["Batch ID","Glucose [mM]","Glutamine [mM]","VCD [cells/mL]","Harvest Time [d]","mAb [mg/L]","Space-Time Profit [$/L/d]",]].head())

condition_columns = ["Glucose [mM]","Glutamine [mM]","VCD [cells/mL]",]
condition_comparison = (optimization_results.groupby(condition_columns, as_index=False).agg(
        Batch_Count=("Batch ID", "count"),
        Mean_mAb_mg_L=("mAb [mg/L]", "mean"),
        SD_mAb_mg_L=("mAb [mg/L]", "std"),
        Mean_Space_Time_Profit=("Space-Time Profit [$/L/d]","mean",),
        SD_Space_Time_Profit=("Space-Time Profit [$/L/d]","std",),
        Maximum_Space_Time_Profit=("Space-Time Profit [$/L/d]","max",),
        Mean_Space_Time_Yield=("Space-Time Yield [mg/L/d]","mean",),))

condition_comparison.to_csv(TABLES_DIR / "Condition_Comparison_Table.csv",index=False,)

feature_columns = ["Glucose [mM]","Glutamine [mM]","VCD [cells/mL]",]
target_column = "Space-Time Profit [$/L/d]"
modeling_data = optimization_results[["Batch ID"] + feature_columns + [target_column]].copy()

X = modeling_data[feature_columns]
y = modeling_data[target_column].astype(float)

groups = modeling_data.groupby(feature_columns,dropna=False,).ngroup()
number_of_groups = groups.nunique()

if number_of_groups < 2:
    raise ValueError("At least two distinct initial-condition groups are required "
        "for grouped cross-validation.")

n_splits = min(5, number_of_groups)
group_cv = GroupKFold(n_splits=n_splits)

models = {"Baseline": DummyRegressor(strategy="mean"),
          "Linear Regression": make_pipeline(StandardScaler(),LinearRegression(),),
          "Polynomial Regression (degree 2)": make_pipeline(StandardScaler(),PolynomialFeatures(degree=2,include_bias=False,),LinearRegression(),),
          "KNN Regression (k=3)": make_pipeline(StandardScaler(),KNeighborsRegressor(n_neighbors=3,weights="distance",),),}

model_rows = []
oof_predictions = {}

for model_name, candidate_model in models.items():
    predictions = cross_val_predict(clone(candidate_model),X,y,groups=groups,cv=group_cv,)
    fold_rmse = -cross_val_score(clone(candidate_model),X,y,groups=groups,cv=group_cv,scoring="neg_root_mean_squared_error",)
    oof_predictions[model_name] = predictions

    model_rows.append(
        {"Model": model_name,"OOF R2": r2_score(y, predictions),"OOF RMSE": root_mean_squared_error(y, predictions),"Mean Fold RMSE": fold_rmse.mean(),
            "SD Fold RMSE": fold_rmse.std(ddof=1)
            if len(fold_rmse) > 1 else 0.0,})

model_comparison = pd.DataFrame(model_rows).sort_values("Mean Fold RMSE")
model_comparison.to_csv(TABLES_DIR / "Model_Comparison.csv",index=False,)

print("\nModel comparison:")
print(model_comparison.to_string(index=False))

non_baseline_models = {
    name: model
    for name, model in models.items()
    if name != "Baseline"}

best_model_name = model_comparison.loc[model_comparison["Model"].isin(non_baseline_models)].sort_values("Mean Fold RMSE").iloc[0]["Model"]
best_model_template = non_baseline_models[best_model_name]

print(f"\nSelected model: {best_model_name}")

y_mab = optimization_results["mAb [mg/L]"].astype(float)
y_harvest = optimization_results["Harvest Time [d]"].astype(float)

if len(X) != len(y_mab) or len(X) != len(y_harvest):
    raise ValueError(
        "The feature data and target data have different numbers of rows.")

final_mab_model = clone(best_model_template)
final_mab_model.fit(X, y_mab)

final_harvest_model = clone(best_model_template)
final_harvest_model.fit(X, y_harvest)

mab_oof_predictions = cross_val_predict(clone(best_model_template),X,y_mab,groups=groups,cv=group_cv,)
harvest_oof_predictions = cross_val_predict(clone(best_model_template),X,y_harvest,groups=groups,cv=group_cv,)

mab_oof_residuals = y_mab.to_numpy() - mab_oof_predictions
harvest_oof_residuals = y_harvest.to_numpy() - harvest_oof_predictions

candidate_conditions = (modeling_data[feature_columns].drop_duplicates().reset_index(drop=True))
candidate_X = candidate_conditions[feature_columns]

mab_predictions = final_mab_model.predict(candidate_X)
harvest_predictions = final_harvest_model.predict(candidate_X)

rng = np.random.default_rng(RANDOM_STATE)

sampled_indices = rng.integers(low=0,high=len(mab_oof_residuals),size=(N_SIMULATIONS, len(candidate_conditions)),)
mab_simulations = (mab_predictions[np.newaxis, :]+ mab_oof_residuals[sampled_indices])
harvest_simulations = (harvest_predictions[np.newaxis, :]+ harvest_oof_residuals[sampled_indices])

mab_simulations = np.maximum(mab_simulations, 0.0)
harvest_simulations = np.maximum(harvest_simulations, 0.0)

culture_costs = (10 * candidate_conditions["Glucose [mM]"].to_numpy()+ 20 * candidate_conditions["Glutamine [mM]"].to_numpy()+ 0.0005 * candidate_conditions["VCD [cells/mL]"].to_numpy())

# Calculate culture value for every simulation.
culture_value_simulations = 1000 * mab_simulations

# Calculate profit for every simulation, including the 1-day downtime.
profit_simulations = (culture_value_simulations - culture_costs[np.newaxis, :]) / (harvest_simulations + 1.0)

# Calculate mAb space-time yield for every simulation.
yield_simulations = mab_simulations / (harvest_simulations + 1.0)

# Calculate the lower, median, and upper quantiles.
# For a 95% interval, these are the 2.5th, 50th, and 97.5th percentiles.
alpha = 1 - PREDICTION_INTERVAL
lower_quantile = alpha / 2
upper_quantile = 1 - alpha / 2


def summarize_simulations(simulations):
    lower = np.quantile(simulations, lower_quantile, axis=0)
    median = np.quantile(simulations, 0.50, axis=0)
    upper = np.quantile(simulations, upper_quantile, axis=0)
    return lower, median, upper


(value_lower,value_median,value_upper,) = summarize_simulations(culture_value_simulations)
(profit_lower,profit_median,profit_upper,) = summarize_simulations(profit_simulations)
(yield_lower,yield_median,yield_upper,) = summarize_simulations(yield_simulations)
(harvest_lower,harvest_median,harvest_upper,) = summarize_simulations(harvest_simulations)


candidate_conditions["Predicted mAb [mg/L]"] = np.median(mab_simulations, axis=0)
candidate_conditions["Culture Cost [$/L]"] = culture_costs

candidate_conditions["Predicted Culture Value [$/L]"] = value_median
candidate_conditions["Culture Value PI Lower [$/L]"] = value_lower
candidate_conditions["Culture Value PI Upper [$/L]"] = value_upper

candidate_conditions["Predicted Harvest Time [d]"] = harvest_median
candidate_conditions["Harvest Time PI Lower [d]"] = harvest_lower
candidate_conditions["Harvest Time PI Upper [d]"] = harvest_upper

candidate_conditions["Predicted Space-Time Profit [$/L/d]"] = profit_median
candidate_conditions["Profit PI Lower [$/L/d]"] = profit_lower
candidate_conditions["Profit PI Upper [$/L/d]"] = profit_upper

candidate_conditions["Predicted Space-Time Yield [mg/L/d]"] = yield_median
candidate_conditions["Yield PI Lower [mg/L/d]"] = yield_lower
candidate_conditions["Yield PI Upper [mg/L/d]"] = yield_upper

# Rank conditions using median simulated space-time profit.
candidate_conditions = candidate_conditions.sort_values("Predicted Space-Time Profit [$/L/d]",ascending=False,).reset_index(drop=True)

candidate_conditions.insert(0,"Predicted Rank",np.arange(1, len(candidate_conditions) + 1),)
candidate_conditions.to_csv(TABLES_DIR / "Model_Based_Optimization.csv",index=False,)

print("\nModel-based comparison of tested conditions:")
print(candidate_conditions.head(10).to_string(index=False))

recommended_condition = candidate_conditions.iloc[0]

print("\nRecommended tested initial condition:")
print(recommended_condition.to_string())

print("\nApproximate 95% prediction interval for culture value: "f"[{recommended_condition['Culture Value PI Lower [$/L]']:.2f}, "f"{recommended_condition['Culture Value PI Upper [$/L]']:.2f}] $/L")
print("Approximate 95% prediction interval for space-time profit: "f"[{recommended_condition['Profit PI Lower [$/L/d]']:.2f}, "f"{recommended_condition['Profit PI Upper [$/L/d]']:.2f}] $/L/d")
print("Approximate 95% prediction interval for space-time yield: "f"[{recommended_condition['Yield PI Lower [mg/L/d]']:.4f}, "f"{recommended_condition['Yield PI Upper [mg/L/d]']:.4f}] mg/L/d")
print("Approximate 95% prediction interval for harvest time: "f"[{recommended_condition['Harvest Time PI Lower [d]']:.2f}, "f"{recommended_condition['Harvest Time PI Upper [d]']:.2f}] days")

recommended_economic_performance = pd.DataFrame({
    "Economic metric": [
        "Culture cost ($/L)",
        "Culture value ($/L)",
        "Space-time profit ($/L/d)",
        "Space-time yield (mg/L/d)",],
    "Estimate": [
        recommended_condition["Culture Cost [$/L]"],
        recommended_condition["Predicted Culture Value [$/L]"],
        recommended_condition["Predicted Space-Time Profit [$/L/d]"],
        recommended_condition["Predicted Space-Time Yield [mg/L/d]"],],
    "95% PI lower": [
        np.nan,  # deterministic value, no interval
        recommended_condition["Culture Value PI Lower [$/L]"],
        recommended_condition["Profit PI Lower [$/L/d]"],
        recommended_condition["Yield PI Lower [mg/L/d]"],],
    "95% PI upper": [
        np.nan,  # deterministic value, no interval
        recommended_condition["Culture Value PI Upper [$/L]"],
        recommended_condition["Profit PI Upper [$/L/d]"],
        recommended_condition["Yield PI Upper [mg/L/d]"],],})

recommended_economic_performance.to_csv(TABLES_DIR / "Recommended_Economic_Performance.csv",index=False,)

print("\nRECOMMENDED ECONOMIC PERFORMANCE")
print(recommended_economic_performance.round(4).to_string(index=False))

analyzer.save_top_batches(optimization_results, n=10)

best_observed_batch = optimization_results.loc[optimization_results["Space-Time Profit [$/L/d]"].idxmax()]

print("\nBest observed batch:")
print(best_observed_batch.to_string())

fig, ax = plt.subplots(figsize=(8, 6))

for model_name, predictions in oof_predictions.items():
    ax.scatter(y,predictions,label=model_name,alpha=0.8,)

minimum_value = min(y.min(), min(p.min() for p in oof_predictions.values()))
maximum_value = max(y.max(), max(p.max() for p in oof_predictions.values()))

ax.plot([minimum_value, maximum_value],[minimum_value, maximum_value],linestyle="--",label="Ideal prediction",)

ax.set_xlabel("Observed Space-Time Profit [$/L/d]")
ax.set_ylabel("Out-of-Fold Predicted Space-Time Profit [$/L/d]")
ax.set_title("Grouped Cross-Validation Model Comparison")
ax.legend()
fig.tight_layout()

fig.savefig(FIGURES_DIR / "Model_Comparison.png",dpi=300,)
plt.close(fig)

plot_data = candidate_conditions.sort_values("Predicted Space-Time Profit [$/L/d]").reset_index(drop=True)
x_positions = np.arange(len(plot_data))

lower_errors = (plot_data["Predicted Space-Time Profit [$/L/d]"] - plot_data["Profit PI Lower [$/L/d]"]).to_numpy()
upper_errors = (plot_data["Profit PI Upper [$/L/d]"]- plot_data["Predicted Space-Time Profit [$/L/d]"]).to_numpy()

fig, ax = plt.subplots(figsize=(10, 6))

ax.errorbar(x_positions,plot_data["Predicted Space-Time Profit [$/L/d]"],yerr=np.vstack([lower_errors, upper_errors]),fmt="D",capsize=3,)

ax.set_xlabel("Tested Initial-Condition Combination (sorted)")
ax.set_ylabel("Predicted Space-Time Profit [$/L/d]")
ax.set_title("Predicted Space-Time Profit with Approximate 95% Prediction Intervals")
fig.tight_layout()

fig.savefig(FIGURES_DIR / "Profit_Prediction_Intervals.png",dpi=300,)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 6))

scatter = ax.scatter(condition_comparison["VCD [cells/mL]"],condition_comparison["Mean_Space_Time_Profit"],c=condition_comparison["Glutamine [mM]"],s=80,)

ax.set_xlabel("Initial VCD [cells/mL]")
ax.set_ylabel("Mean Observed Space-Time Profit [$/L/d]")
ax.set_title("Observed Performance of Tested Initial Conditions")

colorbar = fig.colorbar(scatter, ax=ax)
colorbar.set_label("Initial Glutamine [mM]")

fig.tight_layout()
fig.savefig(FIGURES_DIR / "Condition_Performance.png",dpi=300,)
plt.close(fig)

print("\nAnalysis complete.")
print(f"Tables saved to: {TABLES_DIR}")
print(f"Figures saved to: {FIGURES_DIR}")


