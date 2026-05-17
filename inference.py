# =========================================================
# INFERENCE + LATENCY TESTING
# =========================================================

import joblib
import time
import numpy as np

# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load("wine_model.pkl")

print("Model loaded successfully!")

# =========================================================
# SAMPLE INPUT
# =========================================================

sample_data = np.array([
    [13.2, 2.77, 2.51, 18.5, 96.6, 1.04, 3.92, 0.32, 1.45, 4.8, 0.89, 2.12, 490]
])

# =========================================================
# LATENCY MEASUREMENT
# =========================================================

start_time = time.time()

prediction = model.predict(sample_data)

end_time = time.time()

latency = (end_time - start_time) * 1000

# =========================================================
# OUTPUT
# =========================================================

print("\nPrediction:", prediction[0])

print(f"Latency: {latency:.4f} ms")