"""
generate_sample_data.py
=======================
Creates SIMULATED temperature and vibration logs for ring-frame spindle
bearings, so that ring_frame_life_predictor.py can be tried out before real
mill data is available.  The numbers are invented, but they follow the way
bearings really wear: a long healthy period, then an accelerating wear-out
phase that ends in failure.

Files written to the data/ folder:
  training_history.csv    60 bearings logged from installation until failure
  current_readings.csv    12 bearings that are still running (to be predicted)
  current_answer_key.csv  the true remaining life of those 12 bearings, so the
                          predictions can be checked (real data has no such file)

Run:  python generate_sample_data.py
"""

# Path builds folder and file names that work on Windows, Linux and macOS alike.
from pathlib import Path

# NumPy: fast maths on whole arrays of numbers, plus a random-number generator.
import numpy as np
# pandas: tables of data (called DataFrames) and reading/writing of CSV files.
import pandas as pd

# One averaged reading is logged per 8-hour shift.
READING_INTERVAL_HOURS = 8
# How many failed bearings the training history contains.
HISTORY_BEARINGS = 60
# How many bearings are still running on the machine.
RUNNING_BEARINGS = 12
# Fixed seed: the "random" numbers are the same on every run, so the files can be reproduced exactly.
SEED = 42
# Folder that receives the CSV files.
DATA_DIR = Path("data")


def simulate_bearing(component_id, rng):
    """Return the complete log of one bearing, from installation to failure."""
    # Total life in hours, drawn from a Weibull distribution (the standard model for bearing life), never below 2,500 h.
    life_hours = max(2500, rng.weibull(2.5) * 8000)
    # The operating hours at which readings are logged: 8, 16, 24, ... up to the failure.
    hours = np.arange(READING_INTERVAL_HOURS, int(life_hours) + 1, READING_INTERVAL_HOURS)
    # Length of the final wear-out phase: the last 600 to 1,500 hours before failure.
    wear_out_hours = rng.uniform(600, 1500)
    # 0 while the bearing is healthy, then rising evenly to 1 during the wear-out phase (np.clip keeps it between 0 and 1).
    progress = np.clip((hours - (life_hours - wear_out_hours)) / wear_out_hours, 0, 1)
    # How sharply the damage accelerates towards the end; different for every bearing.
    acceleration = rng.uniform(3, 6)
    # Damage from 0 (healthy) to 1 (failed), growing exponentially: slowly at first, quickly at the end.
    damage = (np.exp(acceleration * progress) - 1) / (np.exp(acceleration) - 1)
    # How early the temperature reacts to damage: an exponent below 1 reacts early, above 1 reacts late.
    heat_exponent = rng.uniform(0.6, 2.0)
    # The same for vibration. Lubrication trouble shows first as heat, surface fatigue shows first as vibration.
    shake_exponent = rng.uniform(0.6, 2.0)
    # Slow seasonal swing of the room temperature: +/- 3 °C over one year (8,760 h), starting at a random time of year.
    season = 3 * np.sin(2 * np.pi * hours / 8760 + rng.uniform(0, 2 * np.pi))
    # Random sensor noise of the thermometer: a different small error (about 0.8 °C) on every reading.
    temperature_noise = rng.normal(0, 0.8, len(hours))
    # Temperature = this bearing's own normal level + season + heating caused by damage + noise.
    temperature = rng.normal(45, 2) + season + rng.uniform(20, 35) * damage ** heat_exponent + temperature_noise
    # Vibration noise becomes up to four times rougher as the damage grows, because a worn bearing runs unevenly.
    vibration_noise = rng.normal(0, 0.06, len(hours)) * (1 + 3 * damage)
    # Vibration = this bearing's own normal level + rise caused by damage + noise.
    vibration = rng.normal(0.8, 0.1) + rng.uniform(4, 7) * damage ** shake_exponent + vibration_noise
    # Put the four columns into one table; every row is one reading.
    return pd.DataFrame({
        # The same bearing name on every row of this log.
        "component_id": component_id,
        # Age of the bearing at the moment of the reading.
        "operating_hours": hours,
        # Bearing temperature in degrees Celsius, rounded like a real sensor would report it.
        "temperature_c": temperature.round(2),
        # Vibration velocity (RMS) in millimetres per second.
        "vibration_mm_s": vibration.round(3),
    })


def main():
    """Simulate all bearings and write the three CSV files."""
    # Random-number generator that starts from the fixed seed.
    rng = np.random.default_rng(SEED)
    # Create the data folder; exist_ok=True means "no error if it is already there".
    DATA_DIR.mkdir(exist_ok=True)

    # Simulate the failed bearings BRG-001 ... BRG-060 (the f-string writes the number with three digits).
    history_logs = [simulate_bearing(f"BRG-{number:03d}", rng) for number in range(1, HISTORY_BEARINGS + 1)]
    # Stack the 60 logs on top of each other and save them as one CSV file (index=False: no extra row-number column).
    pd.concat(history_logs).to_csv(DATA_DIR / "training_history.csv", index=False)

    # Empty list that will collect the visible logs of the running bearings.
    running_logs = []
    # Empty list that will collect their true remaining life.
    answer_key = []
    # How far from failure each running bearing is "today": from 20 to 2,000 hours, spread evenly on a log scale and
    # then shuffled, so that the demonstration contains healthy, worn and nearly failed bearings in no particular order.
    hours_left_today = rng.permutation(np.geomspace(20, 2000, RUNNING_BEARINGS))
    # enumerate() numbers the bearings 101, 102, ... and hands out one "hours left" value to each.
    for number, hours_left in enumerate(hours_left_today, start=101):
        # Simulate the whole life first; we then hide the part that lies in the future.
        full_log = simulate_bearing(f"BRG-{number:03d}", rng)
        # The hour at which this bearing will fail = the last row of its full log.
        failure_hour = full_log["operating_hours"].iloc[-1]
        # Keep only the readings taken up to "today".
        log_so_far = full_log[full_log["operating_hours"] <= failure_hour - hours_left]
        # Store the visible part of the log.
        running_logs.append(log_so_far)
        # Store the true remaining life: failure hour minus the hour of the last visible reading.
        answer_key.append({
            # Name of the bearing.
            "component_id": f"BRG-{number:03d}",
            # Hours that really remain after the last visible reading.
            "true_rul_hours": failure_hour - log_so_far["operating_hours"].iloc[-1],
        })

    # Save the logs of the running bearings: this is the file the predictor will be asked about.
    pd.concat(running_logs).to_csv(DATA_DIR / "current_readings.csv", index=False)
    # Save the answer key separately; the predictor never reads this file.
    pd.DataFrame(answer_key).to_csv(DATA_DIR / "current_answer_key.csv", index=False)
    # Tell the user what was created.
    print(f"Wrote {HISTORY_BEARINGS} failure histories and {RUNNING_BEARINGS} running bearings to the '{DATA_DIR}' folder.")


# Run main() only when this file is started directly, not when another program imports it.
if __name__ == "__main__":
    # Create the sample files.
    main()
