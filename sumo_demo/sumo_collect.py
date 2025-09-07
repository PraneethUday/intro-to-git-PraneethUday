"""
sumo_collect.py
Run this AFTER you generated network.net.xml with netconvert.
Requires: pip install eclipse-sumo
Run: python sumo_collect.py
"""

import os
import csv
import time
import traci
import random

# ---------------- Config ----------------
SUMO_BINARY = "sumo-gui"        # or "sumo-gui" for visualization
NET_FILE = "network.net.xml"
OUTPUT_CSV = "mobility_traces.csv"

SIM_STEPS = 600             # total simulation steps (seconds)
spawn_period = 3            # spawn a vehicle every N steps
vehicle_speed = 13.0        # desired speed (m/s) for new vehicles (approx 47 km/h)
route_id = "r0"

# On Windows, if sumo not in PATH, set full path, e.g.:
# SUMO_BINARY = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe"


# ---------------- Helper to start SUMO ----------------
def start_sumo():
    if not os.path.exists(NET_FILE):
        raise FileNotFoundError(f"{NET_FILE} not found. Generate network with netconvert first.")

    sumo_cmd = [SUMO_BINARY, "-n", NET_FILE, "--step-length", "1.0"]  # 1.0s step
    # if you want GUI visualization, use "sumo-gui" as SUMO_BINARY
    print("Starting SUMO with command:", " ".join(sumo_cmd))
    traci.start(sumo_cmd)


# ---------------- Main simulation & logging ----------------
def run_and_collect():
    # CSV header
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "veh_id", "x", "y", "speed", "angle"])

    # Start SUMO
    start_sumo()

    # create route r0 using edge AB
    try:
        traci.route.add(route_id, ["AB"])
    except Exception as e:
        # If route already exists, ignore
        print("route.add:", e)

    veh_index = 0
    for step in range(SIM_STEPS):
        # spawn a new vehicle every spawn_period steps
        if step % spawn_period == 0:
            vid = f"veh_{veh_index}"
            # add vehicle that will be inserted now
            try:
                traci.vehicle.add(vehID=vid, routeID=route_id, typeID="car")
                # optionally set desired speed
                traci.vehicle.setSpeed(vid, vehicle_speed)  # set immediate speed
            except Exception as e:
                print("vehicle.add error:", e)
            veh_index += 1

        # advance the simulation by one step
        traci.simulationStep()

        # iterate over current vehicles and record data
        ids = traci.vehicle.getIDList()
        if ids:
            rows = []
            for vid in ids:
                x, y = traci.vehicle.getPosition(vid)
                speed = traci.vehicle.getSpeed(vid)
                angle = traci.vehicle.getAngle(vid)
                rows.append([step, vid, x, y, speed, angle])

            # append to CSV
            with open(OUTPUT_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(rows)

        # (optional) small sleep to slow down headless runs if you like
        # time.sleep(0.001)

    # done
    traci.close()
    print(f"Simulation finished. Mobility traces saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    run_and_collect()
