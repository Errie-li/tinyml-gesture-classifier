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

def gen_swipe(direction="left"):
    t = np.linspace(0, DURATION, SAMPLES)
    burst_time = np.random.uniform(0.4, 0.9)
    burst_width = np.random.uniform(0.04, 0.08)
    amplitude = np.random.uniform(1.5, 2.5)
    burst = np.exp(-((t - burst_time)**2) / burst_width) * amplitude
    
    if direction == "left":
        ax = -burst + random_noise(0.05)
        ay = -0.3 * burst + random_noise(0.05)
        gz_amplitude = np.random.uniform(2.5, 3.5)
        gz_burst = np.exp(-((t - burst_time)**2) / (burst_width * 1.2)) * gz_amplitude
        gz = gz_burst + random_noise(0.08)
        gx = 0.4 * burst + random_noise(0.05)
        
    else:  
        ax = burst + random_noise(0.05)
        ay = 0.3 * burst + random_noise(0.05)
        
        gz_amplitude = np.random.uniform(2.5, 3.5)
        gz_burst = np.exp(-((t - burst_time)**2) / (burst_width * 1.2)) * gz_amplitude
        gz = -gz_burst + random_noise(0.08)
        
        gx = -0.4 * burst + random_noise(0.05)
    
    az = 1.0 + random_noise(0.03)
    gy = random_noise(0.03)
    
    data = np.vstack([ax, ay, az, gx, gy, gz]).T
    
    scale = np.random.uniform(0.8, 1.2)
    data[:, :3] = data[:, :3] * scale  
    data[:, 3:] = data[:, 3:] * scale  
    
    return random_time_shift(data, max_shift=5)

def gen_circle():
    
generators = {
    "idle": gen_idle,
    "swipe_left": lambda: gen_swipe("left"),
    "swipe_right": lambda: gen_swipe("right"),
    "circle": gen_circle
}
   

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