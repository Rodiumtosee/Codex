"""
ring_frame_life_predictor.py
============================
Predicts how much life is left in a ring-frame component (for example a
spindle bearing) from its temperature and vibration readings, and turns the
prediction into a preventive-maintenance work list.

ALGORITHM: Random Forest Regression
  A random forest is a collection of decision trees.  Each tree learns, from
  components that have already failed, rules such as "if the vibration rose by
  more than 0.4 mm/s during the last week and the temperature is unsteady,
  then about 150 hours of life were left".  Every tree is trained on a
  different random sample of the history and may only use a random half of the
  measurements at each decision, so the trees make different mistakes.
  The middle answer of all trees (the median) is a stable prediction, and the
  spread between the trees shows how sure the model is.

STEPS PERFORMED
  1. Load the run-to-failure history and analyse the lifetimes.
  2. Train the forest on 75 % of the failed components.
  3. Check it on the other 25 %, which it has never seen.
  4. Predict the Remaining Useful Life (RUL) of the components that are still
     running and give each one a status and a maintenance action.

Run:  python ring_frame_life_predictor.py
      python ring_frame_life_predictor.py --history failures.csv --current machine.csv
"""

# argparse reads options typed after the program name on the command line, for example --history failures.csv.
import argparse
# Path builds folder and file names that work on Windows, Linux and macOS alike.
from pathlib import Path

# matplotlib draws charts.
import matplotlib
# "Agg" draws straight into image files, so the program also works on a computer without a screen.
matplotlib.use("Agg")
# pyplot is the part of matplotlib that creates figures; it has to be imported after the line above.
import matplotlib.pyplot as plt
# NumPy: fast maths on whole arrays of numbers.
import numpy as np
# pandas: tables of data (called DataFrames), grouping, rolling windows and CSV files.
import pandas as pd
# The random-forest algorithm for predicting a number (predicting a number is called regression).
from sklearn.ensemble import RandomForestRegressor
# Mean absolute error: the average size of the prediction error, in the unit of the prediction (hours).
from sklearn.metrics import mean_absolute_error
# Helper that splits a list at random into a training part and a testing part.
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------------------------------------
# SETTINGS - adjust these to your own machine and data
# ---------------------------------------------------------------------------------------------------------
# The sensor columns of the CSV files (left) and the chart title of each (right). Add a line to add a sensor.
SENSORS = {
    # Bearing temperature in degrees Celsius.
    "temperature_c": "Temperature, °C",
    # Vibration velocity (RMS) in millimetres per second.
    "vibration_mm_s": "Vibration, mm/s",
}
# Number of readings in one rolling window: 21 readings = 7 days when one reading is logged per 8-hour shift.
WINDOW = 21
# Prediction horizon in hours (30 days). A healthy component shows no sign of a failure that is further away
# than this, so every longer remaining life is treated as "720 hours or more".
RUL_CAP = 720
# At or below this cautious remaining life (10 days) a component gets the status WARNING.
WARNING_HOURS = 240
# At or below this cautious remaining life (3 days) a component gets the status CRITICAL.
CRITICAL_HOURS = 72
# Hours the machine runs per calendar day (24 for a three-shift mill); used to convert hours into days.
HOURS_PER_DAY = 24
# Share of the failed components that is held back to test the model.
TEST_SHARE = 0.25
# Fixed seed: the random choices (test split, forest) are the same on every run, so results can be repeated.
SEED = 42
# What to do for each status, listed from best to worst.
ACTIONS = {
    # The cautious estimate is above WARNING_HOURS.
    "HEALTHY": "No action - continue routine monitoring",
    # The cautious estimate is between CRITICAL_HOURS and WARNING_HOURS.
    "WARNING": "Order the spare and replace at the next planned stop",
    # The cautious estimate is at or below CRITICAL_HOURS.
    "CRITICAL": "Replace at the next doff - failure expected within days",
}
# Names of the eight model inputs: four statistics for each sensor, for example "vibration_mm_s_trend".
FEATURES = [f"{sensor}_{statistic}" for sensor in SENSORS for statistic in ("mean", "std", "max", "trend")]
# Colours of the chart as hex codes; the blue/orange pair stays distinguishable for colour-blind readers.
COLOURS = {
    # Off-white surface behind the plots.
    "background": "#fcfcfb",
    # Near-black for titles.
    "text": "#0b0b0b",
    # Dark grey for axis numbers and small labels.
    "label": "#52514e",
    # Mid grey for the individual raw readings.
    "raw": "#898781",
    # Pale grey for grid lines.
    "grid": "#e1e0d9",
    # First data colour: the rolling average and the true remaining life.
    "blue": "#2a78d6",
    # Second data colour: the predicted remaining life.
    "orange": "#eb6834",
    # Line that marks the WARNING limit.
    "warning": "#fab219",
    # Line that marks the CRITICAL limit.
    "critical": "#d03b3b",
}


def load_readings(path):
    """Read a CSV file of sensor readings, check it and put it in time order."""
    # Read the file into a table; every row is one reading.
    readings = pd.read_csv(path)
    # The columns the program cannot work without (the * unpacks the sensor names into the list).
    required = ["component_id", "operating_hours", *SENSORS]
    # Collect the names of required columns that are absent from the file.
    missing = [column for column in required if column not in readings.columns]
    # If any column is absent ...
    if missing:
        # ... stop now with a clear message instead of failing later with a confusing one.
        raise ValueError(f"{path} has no column(s) named {missing}")
    # Remove rows in which a required value is empty, for example because a sensor dropped out.
    readings = readings.dropna(subset=required)
    # Sort by component and then by age, and renumber the rows 0, 1, 2, ...
    return readings.sort_values(["component_id", "operating_hours"]).reset_index(drop=True)


def add_features(readings):
    """Describe the recent behaviour of each sensor with four numbers per reading (the model inputs)."""
    # Do the same calculation for the temperature column and then for the vibration column.
    for sensor in SENSORS:
        # Split the column by component, so that a window never mixes the readings of two components.
        by_component = readings.groupby("component_id")[sensor]
        # LEVEL: average of the last WINDOW readings; averaging removes most of the sensor noise. "lambda s:"
        # is a small unnamed function that transform() runs on the series s of each component, and
        # min_periods=1 lets a new component start with fewer readings than a full window.
        readings[f"{sensor}_mean"] = by_component.transform(lambda s: s.rolling(WINDOW, min_periods=1).mean())
        # VARIABILITY: standard deviation of the last WINDOW readings; a worn component runs unsteadily.
        # A single reading has no deviation, so that one gap per component is filled with 0.
        readings[f"{sensor}_std"] = by_component.transform(lambda s: s.rolling(WINDOW, min_periods=1).std()).fillna(0)
        # PEAK: the highest of the last WINDOW readings.
        readings[f"{sensor}_max"] = by_component.transform(lambda s: s.rolling(WINDOW, min_periods=1).max())
        # TREND: the level now minus the level one window earlier (7 days with the settings above) = how fast
        # the sensor is rising. It is unknown during the first window of a log and is filled with 0 there.
        readings[f"{sensor}_trend"] = readings.groupby("component_id")[f"{sensor}_mean"].diff(WINDOW).fillna(0)
    # Hand the table back with its eight new columns.
    return readings


def add_rul(history):
    """Label every historical reading with the life that was really left at that moment."""
    # For every row: the age at which its component failed = the largest age in that component's log.
    failure_hour = history.groupby("component_id")["operating_hours"].transform("max")
    # Remaining Useful Life (RUL) = age at failure minus age at this reading.
    history["true_rul_hours"] = failure_hour - history["operating_hours"]
    # What the model has to learn: the same value, but never more than the horizon RUL_CAP.
    history["target_rul_hours"] = history["true_rul_hours"].clip(upper=RUL_CAP)
    # Hand the table back with its two new columns.
    return history


def status_of(rul_hours):
    """Translate remaining life in hours into HEALTHY, WARNING or CRITICAL."""
    # The two tests, most urgent first: is the life at or below the critical limit / at or below the warning limit?
    tests = [rul_hours <= CRITICAL_HOURS, rul_hours <= WARNING_HOURS]
    # np.select returns the label of the first test that is true, or "HEALTHY" if neither is true.
    return np.select(tests, ["CRITICAL", "WARNING"], default="HEALTHY")


def predict_rul(model, table):
    """Add the predicted remaining life and the resulting status to every row of the table."""
    # The eight feature columns as a plain array of numbers, the form the individual trees work with.
    inputs = table[FEATURES].to_numpy()
    # Ask each tree of the forest separately: one row of answers per tree, one column per reading.
    answers = np.array([tree.predict(inputs) for tree in model.estimators_])
    # For every reading, line the answers of the trees up from shortest to longest life and read off the
    # values found 10 %, 50 % and 90 % of the way along (the 10th, 50th and 90th percentile).
    percentiles = np.percentile(answers, [10, 50, 90], axis=0)
    # CAUTIOUS estimate: the 10th percentile - nine trees out of ten predict a longer life than this.
    table["cautious_rul_hours"] = percentiles[0]
    # BEST estimate: the 50th percentile (the median) - half of the trees say less, half say more.
    table["predicted_rul_hours"] = percentiles[1]
    # OPTIMISTIC estimate: the 90th percentile - only one tree in ten predicts a longer life than this.
    table["optimistic_rul_hours"] = percentiles[2]
    # The status is taken from the cautious estimate: replacing a little early costs far less than a breakdown.
    table["status"] = status_of(table["cautious_rul_hours"])
    # Hand the table back with its four new columns.
    return table


def check_model(model, test):
    """Print how well the model predicts components it has never seen, and return their predictions."""
    # Predict the remaining life for every reading of the unseen test components.
    test = predict_rul(model, test)
    # Readings from the wear-out phase (less than the horizon left): here the prediction is a real countdown.
    wear_out = test[test["true_rul_hours"] < RUL_CAP]
    # Readings from the last days before the failure, where accuracy matters most.
    last_days = test[test["true_rul_hours"] <= WARNING_HOURS]
    # Average size of the error of the best estimate during the wear-out phase ...
    error_wear_out = mean_absolute_error(wear_out["target_rul_hours"], wear_out["predicted_rul_hours"])
    # ... and during the last days only.
    error_last_days = mean_absolute_error(last_days["target_rul_hours"], last_days["predicted_rul_hours"])
    # Share of wear-out readings whose true life lies between the cautious and the optimistic estimate.
    inside = wear_out["target_rul_hours"].between(wear_out["cautious_rul_hours"], wear_out["optimistic_rul_hours"]).mean()
    # Share of wear-out readings whose true life was shorter than even the cautious estimate.
    below = (wear_out["target_rul_hours"] < wear_out["cautious_rul_hours"]).mean()
    # For each test component: the life that was left at its last HEALTHY reading. From that moment until the
    # failure the component was flagged without interruption, so this is the warning time the mill would have had.
    warning_time = test[test["status"] == "HEALTHY"].groupby("component_id")["true_rul_hours"].min()
    # Readings with more than the whole horizon of life left: the model should call every one of them HEALTHY.
    far_from_failure = test["true_rul_hours"] >= RUL_CAP
    # False alerts: readings of that kind which were flagged WARNING or CRITICAL anyway.
    false_alerts = (far_from_failure & (test["status"] != "HEALTHY")).sum()
    # Print the error of the wear-out phase (":.0f" shows a number without decimals).
    print(f"  Average error of the predicted life in the last {RUL_CAP} h before failure: {error_wear_out:.0f} h")
    # Print the error of the last days.
    print(f"  Average error of the predicted life in the last {WARNING_HOURS} h before failure: {error_last_days:.0f} h")
    # Print how often the truth fell inside the range of the trees (":.0%" shows a share as a percentage).
    print(f"  True life inside the cautious-to-optimistic range: {inside:.0%} of the readings in the last {RUL_CAP} h")
    # Print how often even the cautious estimate was too high.
    print(f"  True life shorter than even the cautious estimate: {below:.0%} of the readings in the last {RUL_CAP} h")
    # Print the warning time: the shortest one is the worst case among the test components.
    print(f"  Warning time before failure: shortest {warning_time.min():.0f} h, median {warning_time.median():.0f} h")
    # Print the number of false alerts next to the number of readings that could have caused one.
    print(f"  False alerts with more than {RUL_CAP} h left: {false_alerts} of {far_from_failure.sum():,} readings")
    # The status each test reading really had, worked out from its true remaining life.
    test["true_status"] = status_of(test["true_rul_hours"])
    # Count the readings for every pair of (true status, status given by the model).
    counts = pd.crosstab(test["true_status"], test["status"], rownames=["true status"], colnames=["model said"])
    # Put rows and columns in the order HEALTHY, WARNING, CRITICAL and write 0 where a pair never occurred.
    counts = counts.reindex(index=list(ACTIONS), columns=list(ACTIONS), fill_value=0)
    # Print a heading for the table ...
    print("  Number of test readings by true status (rows) and by the status the model gave (columns):")
    # ... and the table itself: numbers on the diagonal are readings for which the model gave the right status.
    print(counts.to_string())
    # Hand the test table back with its predictions, for the chart.
    return test


def save_chart(component, path):
    """Save a chart of one test component: its sensor readings and its true and predicted remaining life."""
    # Style for the whole chart. Each entry below is "setting name": value.
    plt.rcParams.update({
        # The same off-white behind the figure and behind each plot.
        "figure.facecolor": COLOURS["background"], "axes.facecolor": COLOURS["background"],
        # Near-black titles and dark-grey axis labels.
        "text.color": COLOURS["text"], "axes.labelcolor": COLOURS["label"],
        # Dark-grey numbers along both axes.
        "xtick.color": COLOURS["label"], "ytick.color": COLOURS["label"],
        # Thin pale grid lines behind the data, and the same pale colour for the axis lines.
        "axes.grid": True, "grid.color": COLOURS["grid"], "grid.linewidth": 0.8, "axes.edgecolor": COLOURS["grid"],
        # No box line at the top and at the right of each plot.
        "axes.spines.top": False, "axes.spines.right": False,
        # Data lines 2 points thick; legends without a frame and with slightly smaller text.
        "lines.linewidth": 2, "legend.frameon": False, "legend.fontsize": 9,
    })
    # One plot per sensor plus one for the remaining life, stacked on top of each other and sharing the time
    # axis; the figure is 10 inches wide and 3.2 inches high per plot.
    figure, plots = plt.subplots(len(SENSORS) + 1, 1, figsize=(10, 3.2 * (len(SENSORS) + 1)), sharex=True)
    # The time axis: age of the component at each reading.
    hours = component["operating_hours"]
    # Fill the sensor plots one by one (zip pairs each plot with a sensor column and its title).
    for plot, (sensor, title) in zip(plots, SENSORS.items()):
        # Every raw reading as a thin grey line, to show the noise the model has to see through.
        plot.plot(hours, component[sensor], color=COLOURS["raw"], linewidth=1, label="Each reading")
        # The rolling average as a thick blue line: this is the "level" feature given to the model.
        plot.plot(hours, component[f"{sensor}_mean"], color=COLOURS["blue"], label=f"Average of the last {WINDOW} readings")
        # Sensor name and unit as the title, at the top left.
        plot.set_title(title, loc="left", fontsize=11)
    # One legend explains the grey and blue lines of all sensor plots. It sits just above the first plot at the
    # right (its lower-right corner is pinned to the plot's upper-right corner), where it cannot cover data.
    plots[0].legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=2)
    # The last plot shows the remaining life.
    life_plot = plots[-1]
    # Pale orange band between the cautious and the optimistic estimate: it is wide where the trees disagree
    # (alpha=0.15 makes the colour 85 % transparent, linewidth=0 removes the outline of the band).
    life_plot.fill_between(hours, component["cautious_rul_hours"], component["optimistic_rul_hours"],
                           color=COLOURS["orange"], alpha=0.15, linewidth=0, label="Cautious to optimistic estimate")
    # Blue line: the life that was really left (capped at the horizon).
    life_plot.plot(hours, component["target_rul_hours"], color=COLOURS["blue"], label="True remaining life")
    # Orange line: the best estimate of the model.
    life_plot.plot(hours, component["predicted_rul_hours"], color=COLOURS["orange"], label="Predicted remaining life")
    # The two alarm limits with their name, value and colour.
    limits = [("WARNING", WARNING_HOURS, COLOURS["warning"]), ("CRITICAL", CRITICAL_HOURS, COLOURS["critical"])]
    # Draw them one after the other.
    for name, limit, colour in limits:
        # Dashed horizontal line at the limit.
        life_plot.axhline(limit, color=colour, linestyle="--", linewidth=1.5)
        # Words that name the line, so that it is identified by text and not by colour alone.
        label = f"{name} at or below {limit} h"
        # Write them just above the line at the left edge of the plot.
        life_plot.text(hours.iloc[0], limit, label, va="bottom", fontsize=9, color=COLOURS["label"])
    # Title of the life plot at the top left.
    life_plot.set_title("Remaining useful life, hours", loc="left", fontsize=11)
    # Its legend just above the plot at the right, like the first one, with the three entries side by side.
    life_plot.legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=3)
    # Label of the shared time axis.
    life_plot.set_xlabel("Operating hours since installation")
    # Write the hours on the time axis with a thousands separator, for example 6,400.
    life_plot.xaxis.set_major_formatter("{x:,.0f}")
    # Headline text of the whole figure, with the name of the component.
    headline = f"Test component {component['component_id'].iloc[0]}: the last {len(component)} readings before its failure"
    # Put the headline in bold at the top left of the figure.
    figure.suptitle(headline, x=0.02, ha="left", fontsize=13, fontweight="bold")
    # Arrange the plots so that titles, legends and labels do not overlap.
    figure.tight_layout()
    # Save the figure as a PNG image with 150 dots per inch.
    figure.savefig(path, dpi=150)
    # Close the figure to release the memory it used.
    plt.close(figure)


def main():
    """Run the whole analysis: from the CSV files to the maintenance report."""
    # Describe the program; argparse builds the --help text from this description and the options below.
    parser = argparse.ArgumentParser(description="Predict the remaining useful life of ring-frame components.")
    # Option 1: file with the complete logs of components that have already failed (used to train the model).
    parser.add_argument("--history", default="data/training_history.csv", help="CSV of components logged until failure")
    # Option 2: file with the logs of the components that are running now (the ones we want predictions for).
    parser.add_argument("--current", default="data/current_readings.csv", help="CSV of components still running")
    # Option 3: folder that receives the report and the chart.
    parser.add_argument("--output", default="output", help="folder for the report and the chart")
    # Read the options the user typed; every option has a default, so all of them may be left out.
    args = parser.parse_args()
    # Turn the folder name into a Path object.
    output_dir = Path(args.output)
    # Create the folder (and any missing parent folders); no error if it already exists.
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- STEP 1: load the failure history and analyse it ----------------
    # Load the history, add the eight features and add the remaining-life labels.
    history = add_rul(add_features(load_readings(args.history)))
    # Life of each failed component = its age at its last reading.
    lifetimes = history.groupby("component_id")["operating_hours"].max()
    # Average sensor readings in each stage of life, in the order HEALTHY, WARNING, CRITICAL.
    stage_levels = history.groupby(status_of(history["true_rul_hours"]))[list(SENSORS)].mean().reindex(list(ACTIONS))
    # Print the heading of this step.
    print("STEP 1 - FAILURE HISTORY")
    # Print the size of the history (":," writes a number with a thousands separator).
    print(f"  Failed components: {len(lifetimes)}   Readings: {len(history):,}")
    # Print the average, shortest and longest life.
    print(f"  Life in hours: average {lifetimes.mean():,.0f}, shortest {lifetimes.min():,}, longest {lifetimes.max():,}")
    # Print the L10 life, the usual yardstick for bearings: the age that 90 % of the components reached.
    print(f"  L10 life (10 % had failed by this age): {lifetimes.quantile(0.10):,.0f} h")
    # Print a heading for the table of stage averages ...
    print(f"  Average readings by stage of life (CRITICAL = last {CRITICAL_HOURS} h, WARNING = {CRITICAL_HOURS}-{WARNING_HOURS} h before failure):")
    # ... and the table itself, rounded to two decimals.
    print(stage_levels.round(2).to_string())

    # ---------------- STEP 2: train the random forest ----------------
    # Split the COMPONENTS (not the rows) at random: 75 % for training, 25 % for testing. Neighbouring readings
    # of one component are almost identical, so a component must be entirely in one part; otherwise the test
    # would be far too easy and would overstate the accuracy.
    train_ids, test_ids = train_test_split(list(lifetimes.index), test_size=TEST_SHARE, random_state=SEED)
    # All readings of the training components.
    train = history[history["component_id"].isin(train_ids)]
    # All readings of the test components (.copy() makes an independent table that we may add columns to).
    test = history[history["component_id"].isin(test_ids)].copy()
    # Set up the forest.
    model = RandomForestRegressor(
        # Number of decision trees; more trees give a steadier result.
        n_estimators=200,
        # At each decision a tree may only choose among a random half of the eight features; this keeps the
        # trees different from each other.
        max_features=0.5,
        # Build the trees on all processor cores at once.
        n_jobs=-1,
        # Make the random choices repeatable.
        random_state=SEED,
    )
    # Train: every tree draws its own random sample of the training readings and learns to predict the target.
    model.fit(train[FEATURES], train["target_rul_hours"])
    # Print the heading of this step (the \n starts it on a new line after an empty line).
    print("\nSTEP 2 - TRAINING")
    # Print what was trained on what.
    print(f"  Random forest of {model.n_estimators} trees trained on {len(train_ids)} components ({len(train):,} readings).")

    # ---------------- STEP 3: check the model on components it has never seen ----------------
    # Print the heading of this step.
    print(f"\nSTEP 3 - CHECK ON {len(test_ids)} COMPONENTS THE MODEL HAS NEVER SEEN")
    # Predict the test components, print the quality figures and keep the predictions.
    test = check_model(model, test)
    # How much each feature contributed to the decisions of the trees (the eight shares add up to 1), largest first.
    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    # Print a heading for the list ...
    print("  Share of each measurement in the model's decisions:")
    # ... and the list itself, rounded to three decimals.
    print(importance.round(3).to_string())
    # Readings of the first test component during its last 2 x RUL_CAP hours of life.
    example = test[(test["component_id"] == test_ids[0]) & (test["true_rul_hours"] <= 2 * RUL_CAP)]
    # Draw them and save the chart in the output folder.
    save_chart(example, output_dir / "example_component.png")

    # ---------------- STEP 4: predict the components that are still running ----------------
    # The check is done, so train once more on ALL failed components: more experience gives a better model.
    model.fit(history[FEATURES], history["target_rul_hours"])
    # Load the logs of the running components and add the same eight features.
    current = add_features(load_readings(args.current))
    # Number of readings available for each running component.
    readings_available = current.groupby("component_id").size()
    # Components with less than two full windows of readings: their trend cannot be calculated properly yet.
    too_short = list(readings_available[readings_available < 2 * WINDOW].index)
    # Print the heading of this step.
    print(f"\nSTEP 4 - MAINTENANCE REPORT FOR {len(readings_available)} RUNNING COMPONENTS (most urgent first)")
    # If there are components with too few readings ...
    if too_short:
        # ... name them, because their prediction rests on incomplete information.
        print(f"  NOTE: fewer than {2 * WINDOW} readings for {too_short} - treat their prediction with care.")
    # Keep the newest reading of every component (its features already summarise the recent past) and predict it.
    latest = predict_rul(model, current.groupby("component_id").tail(1).copy())
    # Convert the cautious estimate from operating hours into calendar days.
    latest["cautious_rul_days"] = latest["cautious_rul_hours"] / HOURS_PER_DAY
    # Look up the maintenance action that belongs to each status.
    latest["action"] = latest["status"].map(ACTIONS)
    # Most urgent component first: lowest cautious estimate, then lowest best estimate, then by name.
    latest = latest.sort_values(["cautious_rul_hours", "predicted_rul_hours", "component_id"])
    # Columns of the report, in reading order.
    report_columns = ["component_id", "operating_hours", *SENSORS, "predicted_rul_hours", "cautious_rul_hours",
                      "optimistic_rul_hours", "cautious_rul_days", "status", "action"]
    # Number of decimals for the calculated columns: whole hours, days with one decimal.
    decimals = {"predicted_rul_hours": 0, "cautious_rul_hours": 0, "optimistic_rul_hours": 0, "cautious_rul_days": 1}
    # Build the report: pick the columns and round them.
    report = latest[report_columns].round(decimals)
    # Save the report as a CSV file that opens in Excel (index=False: no extra row-number column).
    report.to_csv(output_dir / "maintenance_report.csv", index=False)
    # Show the report on the screen.
    print(report.to_string(index=False))
    # Remind the reader what a value at the horizon means.
    print(f"  A predicted life of {RUL_CAP} h means 'no wear visible yet: {RUL_CAP} h or more'.")
    # Say where the two output files are.
    print(f"  Saved: {output_dir / 'maintenance_report.csv'} and {output_dir / 'example_component.png'}")


# Run main() only when this file is started directly, not when another program imports it.
if __name__ == "__main__":
    # Start the analysis.
    main()
