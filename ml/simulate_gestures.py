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

def random_scale(min_scale=0.7, max_scale=1.3):
    return np.random.uniform(min_scale, max_scale)

def random_noise(scale=0.05):
    return scale * np.random.randn(SAMPLES)

def random_time_shift(data, max_shift=10):
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(data, shift, axis=0)

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
    
    burst_time = np.random.uniform(0.4, 0.8)
    burst_width = np.random.uniform(0.03, 0.06)
    amplitude = np.random.uniform(1.2, 1.8)
    
    burst = np.exp(-((t - burst_time)**2) / burst_width) * amplitude
    
    if direction == "left":
        ax = -burst + random_noise(0.05)
    else:
        ax = burst + random_noise(0.05)
    
    gyro_freq = np.random.uniform(4, 6)
    gyro_amp = np.random.uniform(0.15, 0.25)
    gx = gyro_amp * np.sin(2 * np.pi * gyro_freq * t) + random_noise(0.02)
    
    ay = random_noise(0.02)
    az = 1.0 + random_noise(0.02)
    gy = random_noise(0.01)
    gz = random_noise(0.01)
    
    data = np.vstack([ax, ay, az, gx, gy, gz]).T
    return random_time_shift(data, max_shift=5)

def gen_circle():
    t = np.linspace(0, DURATION, SAMPLES)
    
    frequency = np.random.uniform(1.2, 1.8) 
    radius = np.random.uniform(0.5, 0.7)
    phase_shift = np.random.uniform(0, 2*np.pi)
    
    ax = radius * np.sin(2 * np.pi * frequency * t + phase_shift) + random_noise(0.05)
    ay = radius * np.cos(2 * np.pi * frequency * t + phase_shift) + random_noise(0.05)
    az = 1.0 + random_noise(0.03)
    
    gyro_scale = np.random.uniform(0.35, 0.45)
    gx = gyro_scale * np.sin(2 * np.pi * frequency * t + phase_shift) + random_noise(0.02)
    gy = gyro_scale * np.cos(2 * np.pi * frequency * t + phase_shift) + random_noise(0.02)
    gz = random_noise(0.01)
    
    data = np.vstack([ax, ay, az, gx, gy, gz]).T
    return random_time_shift(data, max_shift=5)

generators = {
    "idle": gen_idle,
    "swipe_left": lambda: gen_swipe("left"),
    "swipe_right": lambda: gen_swipe("right"),
    "circle": gen_circle
}

#the csv stuff 
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