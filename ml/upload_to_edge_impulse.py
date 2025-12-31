import os
import subprocess
import glob

DATA_DIR = "data/raw"
GESTURES = ["idle", "swipe_left", "swipe_right", "circle"]

print("Logging into Edge Impulse...")
subprocess.run(["edge-impulse-uploader", "--clean"], check=False)

for gesture in GESTURES:
    pattern = os.path.join(DATA_DIR, f"{gesture}_*.csv")
    files = sorted(glob.glob(pattern))
    
    for csv_file in files:
        cmd = [
            "edge-impulse-uploader",
            "--category", "split",
            "--label", gesture,
            csv_file
        ]
        
        print(f"Uploading {os.path.basename(csv_file)} as '{gesture}'...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error uploading {csv_file}")
            print(result.stderr)

print("\nUpload complete!")