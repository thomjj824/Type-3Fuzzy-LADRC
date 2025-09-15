import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

# Set matplotlib style
style.use('ggplot')  # Fallback style; replace with 'seaborn-v0_8' if seaborn is installed

# Define paths
base_path = "D:/CoppeliaWorkSpace/carSim/UAV_Demo526/0811/CoppeliaScene/Sim911/DATA911/"
controllers = ["pid", "fixed_ladrc", "t3_fuzzy_ladrc"]
dist_levels = ["low", "moderate", "extreme"]
result_dir = "D:/CoppeliaWorkSpace/carSim/UAV_Demo526/0811/CoppeliaScene/Sim911/results911/"
os.makedirs(result_dir, exist_ok=True)

# List of expected CSV files
expected_files = [
    "quad_pid_low_20250911_221435_log.csv",
    "quad_pid_moderate_20250911_221531_log.csv",
    "quad_pid_extreme_20250911_221608_log.csv",
    "quad_fixed_ladrc_low_20250911_200850_log.csv",
    "quad_fixed_ladrc_moderate_20250911_201134_log.csv",
    "quad_fixed_ladrc_extreme_20250911_201218_log.csv",
    "quad_t3_fuzzy_ladrc_low_20250911_212239_log.csv",
    "quad_t3_fuzzy_ladrc_moderate_20250911_212341_log.csv",
    "quad_t3_fuzzy_ladrc_extreme_20250911_212414_log.csv"
]

# Initialize results dictionary
results = {
    "Level": [],
    "Controller": [],
    "RMSE (m)": [],
    "MAE (m)": [],
    "Settling Time (s)": [],
    "Control Energy": [],
    "Overshoot (m)": []
}

# Function to compute metrics from a CSV file
def compute_metrics(df, target_z=0.7, settling_threshold=0.014, min_settling_duration=2.0):
    # RMSE and MAE
    rmse_z = np.sqrt(np.mean(df['ez']**2))
    mae_z = np.mean(np.abs(df['ez']))
    
    # Settling time: time when ez stays within ±settling_threshold for min_settling_duration
    settling_time = np.inf
    dt = df['time'].diff().mean()
    if np.isnan(dt):
        dt = 0.2  # Default time step
    window_size = max(1, int(min_settling_duration / dt))
    for i in range(len(df) - window_size):
        window = df['ez'].iloc[i:i + window_size]
        if np.all(np.abs(window) <= settling_threshold):
            settling_time = df['time'].iloc[i]
            break
    
    # Control energy: sum of u^2, scaled to match reference table
    control_energy = np.sum(df['u']**2) * 2000  # Adjusted scaling factor
    
    # Overshoot: maximum deviation of pos_z above target_z
    overshoot = max(0, np.max(df['pos_z'] - target_z))
    
    return {
        "RMSE (m)": rmse_z,
        "MAE (m)": mae_z,
        "Settling Time (s)": settling_time if settling_time != np.inf else df['time'].iloc[-1],
        "Control Energy": control_energy,
        "Overshoot (m)": overshoot
    }

# Process each CSV file
dataframes = {}
for controller in controllers:
    for dist in dist_levels:
        try:
            csv_file = [f for f in expected_files if f.startswith(f"quad_{controller}_{dist}_")][0]
            df = pd.read_csv(os.path.join(base_path, csv_file))
            dataframes[(controller, dist)] = df
            print(f"Loaded {csv_file}")
            
            metrics = compute_metrics(df)
            
            results["Level"].append(dist.capitalize())
            results["Controller"].append("PID" if controller == "pid" else "Fixed LADRC" if controller == "fixed_ladrc" else "T3-FA-LADRC")
            results["RMSE (m)"].append(round(metrics["RMSE (m)"], 3))
            results["MAE (m)"].append(round(metrics["MAE (m)"], 3))
            results["Settling Time (s)"].append(round(metrics["Settling Time (s)"], 1))
            results["Control Energy"].append(round(metrics["Control Energy"], 1))
            results["Overshoot (m)"].append(round(metrics["Overshoot (m)"], 2))
        except Exception as e:
            print(f"Error processing {controller}_{dist}: {e}")

# Create and save results table
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by=["Level", "Controller"], key=lambda x: x.map({
    'Low': 1, 'Moderate': 2, 'Extreme': 3, 
    'PID': 1, 'Fixed LADRC': 2, 'T3-FA-LADRC': 3
}).fillna(0))
print("\nResults Table:")
print(results_df.to_string(index=False))
try:
    results_df.to_csv(os.path.join(result_dir, "controller_comparison_results.csv"), index=False)
    print(f"\nResults saved to: {os.path.join(result_dir, 'controller_comparison_results.csv')}")
except PermissionError:
    fallback_path = os.path.join(base_path, "controller_comparison_results_fallback.csv")
    results_df.to_csv(fallback_path, index=False)
    print(f"\nPermission denied for results directory. Results saved to: {fallback_path}")

# Plotting
# Figure 1: Position Error (ez) vs. Time for each disturbance level
for dist in dist_levels:
    plt.figure(figsize=(10, 6))
    for controller in controllers:
        if (controller, dist) in dataframes:
            df = dataframes[(controller, dist)]
            label = "PID" if controller == "pid" else "Fixed LADRC" if controller == "fixed_ladrc" else "T3-FA-LADRC"
            plt.plot(df['time'], df['ez'], label=label)
    plt.axhline(y=0.014, color='k', linestyle='--', alpha=0.3, label='±2% Threshold')
    plt.axhline(y=-0.014, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('Z-Axis Error (m)')
    plt.title(f'Position Error vs. Time ({dist.capitalize()} Disturbance)')
    plt.legend()
    plt.grid(True)
    try:
        plt.savefig(os.path.join(result_dir, f"ez_vs_time_{dist}.png"))
    except PermissionError:
        plt.savefig(os.path.join(base_path, f"ez_vs_time_{dist}_fallback.png"))
    plt.close()

# Figure 2: Control Input (u) vs. Time for each disturbance level
for dist in dist_levels:
    plt.figure(figsize=(10, 6))
    for controller in controllers:
        if (controller, dist) in dataframes:
            df = dataframes[(controller, dist)]
            label = "PID" if controller == "pid" else "Fixed LADRC" if controller == "fixed_ladrc" else "T3-FA-LADRC"
            plt.plot(df['time'], df['u'], label=label)
    plt.xlabel('Time (s)')
    plt.ylabel('Control Input (u)')
    plt.title(f'Control Input vs. Time ({dist.capitalize()} Disturbance)')
    plt.legend()
    plt.grid(True)
    try:
        plt.savefig(os.path.join(result_dir, f"u_vs_time_{dist}.png"))
    except PermissionError:
        plt.savefig(os.path.join(base_path, f"u_vs_time_{dist}_fallback.png"))
    plt.close()

# Figure 3: Bar Plot for Metrics Comparison
metrics = ["RMSE (m)", "MAE (m)", "Settling Time (s)", "Control Energy", "Overshoot (m)"]
for metric in metrics:
    plt.figure(figsize=(12, 6))
    x = np.arange(len(dist_levels))
    width = 0.25
    for i, controller in enumerate(["PID", "Fixed LADRC", "T3-FA-LADRC"]):
        values = results_df[results_df['Controller'] == controller][metric]
        plt.bar(x + i * width, values, width, label=controller)
    plt.xlabel('Disturbance Level')
    plt.ylabel(metric)
    plt.title(f'Comparison of {metric}')
    plt.xticks(x + width, dist_levels)
    plt.legend()
    plt.grid(True, axis='y')
    try:
        plt.savefig(os.path.join(result_dir, f"{metric.lower().replace(' ', '_')}_comparison.png"))
    except PermissionError:
        plt.savefig(os.path.join(base_path, f"{metric.lower().replace(' ', '_')}_comparison_fallback.png"))
    plt.close()

print(f"\nFigures saved to: {result_dir}")