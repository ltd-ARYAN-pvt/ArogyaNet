
import tensorflow as tf
from tensorflow.keras import layers, Model, Input

from architecture import (
    plain_separable_block,
    residual_block,
)

def standard_block(x, filters, strides=1, name="standard"):
    """Two standard convolutions; no residual connection."""

    x = layers.Conv2D(
        filters, 3, strides=strides,
        padding="same", use_bias=False,
        name=f"{name}_conv1"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(
        filters, 3, padding="same",
        use_bias=False, name=f"{name}_conv2"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    return x


def plain_separable_block(x, filters, strides=1, name="plain"):
    """Two separable convolutions; no residual connection or SE."""

    x = layers.SeparableConv2D(
        filters, 3, strides=strides,
        padding="same", use_bias=False,
        name=f"{name}_sepconv1"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.SeparableConv2D(
        filters, 3, padding="same",
        use_bias=False, name=f"{name}_sepconv2"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    return x



def multi_scale_block(x, filters=192, name="multi_scale"):
    """Parallel 3x3 and 5x5 depthwise-separable convolutions."""

    branch_3 = layers.SeparableConv2D(
        filters // 2, 3,
        padding="same",
        use_bias=False,
        name=f"{name}_3x3"
    )(x)

    branch_3 = layers.BatchNormalization()(branch_3)
    branch_3 = layers.Activation("relu")(branch_3)

    branch_5 = layers.SeparableConv2D(
        filters // 2, 5,
        padding="same",
        use_bias=False,
        name=f"{name}_5x5"
    )(x)

    branch_5 = layers.BatchNormalization()(branch_5)
    branch_5 = layers.Activation("relu")(branch_5)

    merged = layers.Concatenate()([branch_3, branch_5])

    merged = layers.Conv2D(
        filters, 1,
        padding="same",
        use_bias=False,
        name=f"{name}_fusion"
    )(merged)

    merged = layers.BatchNormalization()(merged)

    channels = int(merged.shape[-1])

    se = layers.GlobalAveragePooling2D()(merged)
    se = layers.Dense(max(channels // 8, 1), activation="relu")(se)
    se = layers.Dense(channels, activation="sigmoid")(se)
    se = layers.Reshape((1, 1, channels))(se)

    merged = layers.Multiply()([merged, se])

    output = layers.Add()([x, merged])
    output = layers.Activation("relu")(output)

    return output


def build_ablation_model(variant, input_shape=(128, 128, 1)):

    if variant not in ["A0", "A1", "A2", "A3", "A4"]:
        raise ValueError(f"Unknown variant: {variant}")

    inputs = Input(shape=input_shape, name="chest_xray_input")

    # Common stem
    x = layers.Conv2D(
        32, 3,
        strides=2,
        padding="same",
        use_bias=False,
        name="stem"
    )(inputs)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    def apply_block(x, filters, strides, name):
        if variant == "A0":
            return standard_block(x, filters, strides, name)

        if variant == "A1":
            return plain_separable_block(x, filters, strides, name)

        if variant == "A2":
            return residual_block(
                x, filters, strides,
                use_se=False, name=name
            )

        if variant in ["A3", "A4"]:
            return residual_block(
                x, filters, strides,
                use_se=True, name=name
            )

    # Stage 1: 64 x 64
    x = apply_block(x, 64, 1, "stage1_block1")
    x = apply_block(x, 64, 1, "stage1_block2")

    # Stage 2: 64 -> 32
    x = apply_block(x, 128, 2, "stage2")

    # Stage 3: 32 -> 16
    x = apply_block(x, 192, 2, "stage3")

    # A4 adds multi-scale feature extraction
    if variant == "A4":
        x = multi_scale_block(x, 192, name="multi_scale")

    # Stage 4: 16 -> 8
    x = apply_block(x, 256, 2, "stage4")

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid",
        name="pneumonia_probability"
    )(x)

    return Model(
        inputs,
        outputs,
        name=f"ArogyaNet_{variant}"
    )


if __name__ == "__main__":

    for variant in ["A0", "A1", "A2", "A3", "A4"]:

        model = build_ablation_model(variant)

        print("\n" + "=" * 45)
        print(variant, model.name)
        print("Parameters:", model.count_params())

        dummy = tf.random.normal((2, 128, 128, 1))
        output = model(dummy)

        print("Output shape:", output.shape)

        assert output.shape == (2, 1)
        assert model.count_params() < 2_000_000

    print("\nALL FIVE ABLATION MODELS PASSED!")