import tensorflow as tf
import time

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

x = tf.random.normal((16, 224, 224, 3))
conv = tf.keras.layers.Conv2D(32, 3, activation="relu")

# Warm-up
for _ in range(3):
    y = conv(x)
    _ = y.numpy()

# Benchmark
times = []

for i in range(10):
    start = time.perf_counter()

    y = conv(x)
    _ = y.numpy()

    times.append(time.perf_counter() - start)

print("Average:", sum(times) / len(times))
print("Minimum:", min(times))
print("Maximum:", max(times))
print("All timings:", times)
