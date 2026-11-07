import tensorflow as tf
import time

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

with tf.device("/GPU:0"):
    x = tf.random.normal((16, 224, 224, 3))

    conv = tf.keras.layers.Conv2D(
        32, kernel_size=3, activation="relu"
    )

    start = time.perf_counter()

    y = conv(x)
    _ = y.numpy()

    print("Output shape:", y.shape)
    print("Execution time:", time.perf_counter() - start)
    print("Device:", y.device)
