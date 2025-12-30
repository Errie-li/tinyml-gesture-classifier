import numpy as np
import os
import csv

OUT_DIR = "data/edge_impulse_augmented"
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLE_RATE = 50
DURATION = 2.0
SAMPLES = int(SAMPLE_RATE * DURATION)
GESTURES = ["idle", "swipe_left", "swipe_right", "circle"]
EXAMPLES_PER_GESTURE = 60

def random_noise(scale=0.05):
    return scale * np.random.randn(SAMPLES)

def random_time_shift(data, max_shift=10):
    shift = np.random.randint(-max_shift, max_shift)
    if shift > 0:
        return np.vstack([data[:shift] * 0, data[:-shift]])
    elif shift < 0:
        return np.vstack([data[-shift:], data[shift:] * 0])
    return data

def gen_idle():
    noise_scale = np.random.uniform(0.005, 0.02)
    ax = noise_scale * np.random.randn(SAMPLES)
    ay = noise_scale * np.random.randn(SAMPLES)
    az = 1.0 + noise_scale * 2 * np.random.randn(SAMPLES)
    
    gx = noise_scale * 0.1 * np.random.randn(SAMPLES)
    gy = noise_scale * 0.1 * np.random.randn(SAMPLES)
    gz = noise_scale * 0.1 * np.random.randn(SAMPLES)
    
    return np.vstack([ax, ay, az, gx, gy, gz]).T


# csv files
for gname, gen in generators.items():
    print(f"Generating {gname}...", end=" ")
    for i in range(EXAMPLES_PER_GESTURE):
        data = gen()
        fname = f"{OUT_DIR}/{gname}.sample{i:03d}.csv"
        
        with open(fname, "w", newline='') as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "accX", "accY", "accZ", "gyrX", "gyrY", "gyrZ"])
            for j, row in enumerate(data):
                timestamp_ms = int(j * (1000 / SAMPLE_RATE))
                w.writerow([timestamp_ms] + row.tolist())
    
    print("done.")

print(f"\nGenerated {len(GESTURES) * EXAMPLES_PER_GESTURE} samples total.")