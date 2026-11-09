
import tensorflow as tf
from tensorflow.keras import layers, Model, Input

from architecture import (
    depthwise_separable_conv,
    residual_se_block,
    squeeze_excitation,
)


def multi_scale_block(x, filters=192, name="multi_scale"):
    """Parallel 3x3 and 5x5 depthwise-separable feature extraction."""

    # Branch 1: smaller spatial patterns
    branch_3 = layers.SeparableConv2D(
        filters // 2, 3, padding="same", use_bias=False,
        name=f"{name}_3x3"
    )(x)
    branch_3 = layers.BatchNormalization()(branch_3)
    branch_3 = layers.Activation("relu")(branch_3)

    # Branch 2: larger spatial patterns
    branch_5 = layers.SeparableConv2D(
        filters // 2, 5, padding="same", use_bias=False,
        name=f"{name}_5x5"
    )(x)
    branch_5 = layers.BatchNormalization()(branch_5)
    branch_5 = layers.Activation("relu")(branch_5)

    # Combine both scales
    merged = layers.Concatenate(name=f"{name}_concat")(
        [branch_3, branch_5]
    )

    merged = layers.Conv2D(
        filters, 1, padding="same", use_bias=False,
        name=f"{name}_fusion"
    )(merged)
    merged = layers.BatchNormalization()(merged)

    # Channel attention
    merged = squeeze_excitation(
        merged, reduction=8, name=f"{name}_se"
    )

    # Residual connection
    shortcut = x
    output = layers.Add(name=f"{name}_add")([shortcut, merged])
    output = layers.Activation("relu", name=f"{name}_out")(output)

    return output


def build_arogyanet(input_shape=(128, 128, 1)):

    inputs = Input(shape=input_shape, name="chest_xray_input")

    # Stem: 128 -> 64
    x = layers.Conv2D(
        32, 3, strides=2, padding="same", use_bias=False,
        name="stem_conv"
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Stage 1: 64 x 64
    x = depthwise_separable_conv(x, 64, name="ds_1")
    x = residual_se_block(x, 64, name="res_se_1")

    # Stage 2: 64 -> 32
    x = residual_se_block(x, 128, strides=2, name="res_se_2")

    # Stage 3: 32 -> 16
    x = residual_se_block(x, 192, strides=2, name="res_se_3")

    # Multi-scale feature extraction: 16 x 16
    x = multi_scale_block(x, 192, name="multi_scale_1")

    # Stage 4: 16 -> 8
    x = residual_se_block(x, 256, strides=2, name="res_se_4")

    # Classification head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(0.35)(x)

    outputs = layers.Dense(
        1, activation="sigmoid",
        name="pneumonia_probability"
    )(x)

    return Model(inputs, outputs, name="ArogyaNet_Wide")


if __name__ == "__main__":

    model = build_arogyanet()
    model.summary()

    total = model.count_params()
    trainable = sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )

    print("\nTotal parameters:", total)
    print("Trainable parameters:", trainable)
    print("Model size estimate (FP32):", round(total * 4 / (1024**2), 3), "MB")

    dummy = tf.random.normal((2, 128, 128, 1))
    prediction = model(dummy)

    print("\nInput shape:", dummy.shape)
    print("Output shape:", prediction.shape)
    print("Sample predictions:", prediction.numpy().flatten())

    assert total < 2_000_000, "Parameter limit exceeded!"
    assert prediction.shape == (2, 1)

    print("\nAROGYANET WIDE MODEL TEST PASSED!")
