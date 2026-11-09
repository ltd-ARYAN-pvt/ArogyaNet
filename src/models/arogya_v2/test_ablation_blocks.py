import tensorflow as tf

from architecture import (
    plain_separable_block,
    residual_block
)

inputs = tf.keras.Input(shape=(64, 64, 32))

a1 = plain_separable_block(
    inputs, 64, strides=2, name="test_plain"
)

a2 = residual_block(
    inputs, 64, strides=2,
    use_se=False, name="test_residual"
)

a3 = residual_block(
    inputs, 64, strides=2,
    use_se=True, name="test_residual_se"
)

for name, output in [
    ("Plain separable", a1),
    ("Residual", a2),
    ("Residual + SE", a3)
]:
    model = tf.keras.Model(inputs, output)
    print(name, output.shape, model.count_params())

    assert output.shape == (None, 32, 32, 64)

print("\nALL ABLATION BLOCK TESTS PASSED!")