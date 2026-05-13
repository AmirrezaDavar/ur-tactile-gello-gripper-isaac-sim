import subprocess
import time
import os

SCRIPT_PATH = os.path.join(os.environ["HOME"], "ur5_simulation/execute_data_collection.sh")

for i in range(1, 300):
    print(f"=== Starting data collection {i} ===")

    try:
        # Run the bash launcher (blocking)
        subprocess.run(["bash", SCRIPT_PATH], check=True)
        print(f"--- Data collection {i} finished successfully ---")

    except subprocess.CalledProcessError as e:
        print(f"Iteration {i} failed with exit code {e.returncode}")
        continue

    time.sleep(3)
