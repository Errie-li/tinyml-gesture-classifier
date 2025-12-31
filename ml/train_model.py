import numpy as np
import os
import csv
from datetime import datetime

OUT_DIR = "data/edge_impulse_ready"
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLE_RATE = 50            
DURATION = 2.0            
SAMPLES = int(SAMPLE_RATE * DURATION)
GESTURES = ["idle", "swipe_left", "swipe_right", "circle"]
EXAMPLES_PER_GESTURE = 60

def gen_idle():
    ax = 0.01 * np.random.randn(SAMPLES)
    ay = 0.01 * np.random.randn(SAMPLES)
    az = 1.0 + 0.02 * np.random.randn(SAMPLES)
    gx = 0.001 * np.random.randn(SAMPLES)
    gy = 0.001 * np.random.randn(SAMPLES)
    gz = 0.001 * np.random.randn(SAMPLES)
    return np.vstack([ax,ay,az,gx,gy,gz]).T

def gen_swipe(direction="left"):
    t = np.linspace(0, DURATION, SAMPLES)
    burst = np.exp(-((t-0.6)**2)/(0.04)) * (1.5) 
    if direction == "left":
        ax = -burst + 0.05*np.random.randn(SAMPLES)
    else:
        ax = burst + 0.05*np.random.randn(SAMPLES)
    gx = 0.2 * np.sin(2*np.pi*5*t) + 0.02*np.random.randn(SAMPLES)
    ay = 0.02*np.random.randn(SAMPLES)
    az = 1.0 + 0.02*np.random.randn(SAMPLES)
    gy = 0.01*np.random.randn(SAMPLES)
    gz = 0.01*np.random.randn(SAMPLES)
    return np.vstack([ax,ay,az,gx,gy,gz]).T

def gen_circle():
    t = np.linspace(0, DURATION, SAMPLES)
    ax = 0.6*np.sin(2*np.pi*1.5*t) + 0.05*np.random.randn(SAMPLES)
    ay = 0.6*np.cos(2*np.pi*1.5*t) + 0.05*np.random.randn(SAMPLES)
    az = 1.0 + 0.03*np.random.randn(SAMPLES)
    gx = 0.4*np.sin(2*np.pi*1.5*t) + 0.02*np.random.randn(SAMPLES)
    gy = 0.4*np.cos(2*np.pi*1.5*t) + 0.02*np.random.randn(SAMPLES)
    gz = 0.01*np.random.randn(SAMPLES)
    return np.vstack([ax,ay,az,gx,gy,gz]).T

generators = {
    "idle": gen_idle,
    "swipe_left": lambda: gen_swipe("left"),
    "swipe_right": lambda: gen_swipe("right"),
    "circle": gen_circle
}

for gname, gen in generators.items():
    for i in range(EXAMPLES_PER_GESTURE):
        data = gen()
        
        
        fname = f"{OUT_DIR}/{gname}.sample{i:03d}.csv"
        
        with open(fname, "w", newline='') as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "accX", "accY", "accZ", "gyrX", "gyrY", "gyrZ"])
            
            for j, row in enumerate(data):
                timestamp_ms = int(j * (1000 / SAMPLE_RATE))  
                w.writerow([timestamp_ms] + row.tolist())

