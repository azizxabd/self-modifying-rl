import os
import pickle
import time
import numpy as np

DEMO_PATH    = "results/demos/demonstrations.pkl"
TARGET_DEMOS = 500

print("=== Demo Collection Progress ===")

while True:
    os.system("cls")
    print("=== Demo Collection Progress ===\n")

    if os.path.exists(DEMO_PATH):
        try:
            with open(DEMO_PATH, "rb") as f:
                demos = pickle.load(f)
            n      = len(demos)
            pct    = n / TARGET_DEMOS * 100
            print(f"Status:     COMPLETE — {n} demos saved")
            print(f"Mean height: {np.mean([d['max_height'] for d in demos]):.4f}m")
            print(f"Mean length: {np.mean([d['length'] for d in demos]):.0f} steps")
            break
        except Exception:
            print("Demo file exists but still being written...")
    else:
        # check log files for partial progress
        log_dir = "results/logs"
        model_dir = "results/models"

        print(f"Demo file not saved yet — collection in progress\n")

        # check if any models exist yet
        if os.path.exists(model_dir):
            models = [f for f in os.listdir(model_dir) if f.endswith(".zip")]
            print(f"Models saved so far: {models if models else 'none yet'}")

        # check tensorboard logs
        tb_dir = os.path.join(log_dir, "tensorboard")
        if os.path.exists(tb_dir):
            seeds = os.listdir(tb_dir)
            print(f"Active training seeds: {seeds if seeds else 'none yet'}")

    print(f"\nLast checked: {time.strftime('%H:%M:%S')}")
    print("Refreshing every 10s... (Ctrl+C to stop)")
    time.sleep(10)