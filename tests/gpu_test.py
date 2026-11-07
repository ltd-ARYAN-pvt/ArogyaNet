import tensorflow as tf
import time

print("TensorFlow:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())

gpus = tf.config.list_physical_devices("GPU")
print("Detected GPUs:", gpus)

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if not gpus:
    raise RuntimeError("GPU not detected!")

with tf.device("/GPU:0"):
    a = tf.random.normal((4096, 4096))
    b = tf.random.normal((4096, 4096))

    start = time.perf_counter()
    c = tf.matmul(a, b)
    _ = c.numpy()

    print("Execution time:", time.perf_counter() - start)
    print("Result shape:", c.shape)
    print("Tensor device:", c.device)

print("GPU execution successful!")
