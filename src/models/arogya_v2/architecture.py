
import tensorflow as tf
from tensorflow.keras import layers


# ==========================================
# 1. DEPTHWISE SEPARABLE CONVOLUTION
# ==========================================

def depthwise_separable_conv(
    x,
    filters,
    kernel_size=3,
    strides=1,
    name="ds_conv"
):
    """
    Depthwise + Pointwise convolution
    followed by Batch Normalization and ReLU.
    """

    x = layers.SeparableConv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        name=f"{name}_separable"
    )(x)

    x = layers.BatchNormalization(
        name=f"{name}_bn"
    )(x)

    x = layers.ReLU(
        name=f"{name}_relu"
    )(x)

    return x


# ==========================================
# 2. SQUEEZE-AND-EXCITATION ATTENTION
# ==========================================

def squeeze_excitation(
    x,
    reduction=4,
    name="se"
):
    """
    Learns channel-wise attention weights.
    """

    channels = x.shape[-1]

    if channels is None:
        raise ValueError("Channel dimension must be known.")

    reduced_channels = max(1, channels // reduction)

    # Squeeze: Global Average Pooling
    se = layers.GlobalAveragePooling2D(
        keepdims=True,
        name=f"{name}_squeeze"
    )(x)

    # Excitation: Learn channel importance
    se = layers.Dense(
        reduced_channels,
        activation="relu",
        name=f"{name}_fc1"
    )(se)

    se = layers.Dense(
        channels,
        activation="sigmoid",
        name=f"{name}_fc2"
    )(se)

    # Scale: Recalibrate input feature maps
    output = layers.Multiply(
        name=f"{name}_scale"
    )([x, se])

    return output


# ==========================================
# 3. RESIDUAL SE BLOCK
# ==========================================

def residual_se_block(
    x,
    filters,
    strides=1,
    reduction=4,
    name="res_se"
):
    """
    Residual block containing:
    - Depthwise separable convolutions
    - Batch normalization
    - SE attention
    - Residual shortcut
    """

    shortcut = x

    # Main branch: First separable convolution
    x = layers.SeparableConv2D(
        filters,
        kernel_size=3,
        strides=strides,
        padding="same",
        use_bias=False,
        name=f"{name}_conv1"
    )(x)

    x = layers.BatchNormalization(
        name=f"{name}_bn1"
    )(x)

    x = layers.ReLU(
        name=f"{name}_relu1"
    )(x)

    # Main branch: Second separable convolution
    x = layers.SeparableConv2D(
        filters,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        name=f"{name}_conv2"
    )(x)

    x = layers.BatchNormalization(
        name=f"{name}_bn2"
    )(x)

    # Channel attention
    x = squeeze_excitation(
        x,
        reduction=reduction,
        name=f"{name}_se"
    )

    # Projection shortcut when dimensions differ
    input_channels = shortcut.shape[-1]

    if input_channels != filters or strides != 1:

        shortcut = layers.Conv2D(
            filters,
            kernel_size=1,
            strides=strides,
            padding="same",
            use_bias=False,
            name=f"{name}_projection"
        )(shortcut)

        shortcut = layers.BatchNormalization(
            name=f"{name}_projection_bn"
        )(shortcut)

    # Residual addition
    x = layers.Add(
        name=f"{name}_add"
    )([x, shortcut])

    x = layers.ReLU(
        name=f"{name}_output"
    )(x)

    return x


def plain_separable_block(x, filters, strides=1, name="plain"):
    """Depthwise separable convolution without residual or SE."""

    x = layers.SeparableConv2D(
        filters,
        3,
        strides=strides,
        padding="same",
        use_bias=False,
        name=f"{name}_sepconv"
    )(x)

    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.Activation("relu", name=f"{name}_relu")(x)

    return x


def residual_block(x, filters, strides=1, use_se=False,
                   reduction=8, name="residual"):
    """Residual block with optional SE attention."""

    shortcut = x

    # Main branch
    y = layers.SeparableConv2D(
        filters, 3,
        strides=strides,
        padding="same",
        use_bias=False,
        name=f"{name}_conv1"
    )(x)

    y = layers.BatchNormalization(name=f"{name}_bn1")(y)
    y = layers.Activation("relu", name=f"{name}_relu1")(y)

    y = layers.SeparableConv2D(
        filters, 3,
        padding="same",
        use_bias=False,
        name=f"{name}_conv2"
    )(y)

    y = layers.BatchNormalization(name=f"{name}_bn2")(y)

    # Optional SE attention
    if use_se:
        channels = int(y.shape[-1])

        se = layers.GlobalAveragePooling2D(
            name=f"{name}_se_gap"
        )(y)

        se = layers.Dense(
            max(channels // reduction, 1),
            activation="relu",
            name=f"{name}_se_reduce"
        )(se)

        se = layers.Dense(
            channels,
            activation="sigmoid",
            name=f"{name}_se_expand"
        )(se)

        se = layers.Reshape((1, 1, channels))(se)
        y = layers.Multiply(name=f"{name}_se_scale")([y, se])

    # Projection shortcut when dimensions differ
    if strides != 1 or int(shortcut.shape[-1]) != filters:

        shortcut = layers.Conv2D(
            filters,
            1,
            strides=strides,
            padding="same",
            use_bias=False,
            name=f"{name}_projection"
        )(shortcut)

        shortcut = layers.BatchNormalization(
            name=f"{name}_projection_bn"
        )(shortcut)

    y = layers.Add(name=f"{name}_add")([shortcut, y])
    y = layers.Activation("relu", name=f"{name}_out")(y)

    return y



# ==========================================
# 4. BLOCK VERIFICATION
# ==========================================

if __name__ == "__main__":

    print("\nAROGYANET BLOCK VERIFICATION")
    print("=" * 45)

    inputs = layers.Input(
        shape=(64, 64, 32)
    )

    # Test 1: Depthwise separable convolution
    ds_output = depthwise_separable_conv(
        inputs,
        filters=64,
        strides=2,
        name="test_ds"
    )

    ds_model = tf.keras.Model(
        inputs,
        ds_output
    )

    print("\n1. Depthwise Separable Convolution")
    print("Input shape:", ds_model.input_shape)
    print("Output shape:", ds_model.output_shape)
    print("Parameters:", ds_model.count_params())

    # Test 2: SE attention
    se_output = squeeze_excitation(
        inputs,
        reduction=4,
        name="test_se"
    )

    se_model = tf.keras.Model(
        inputs,
        se_output
    )

    print("\n2. Squeeze-and-Excitation")
    print("Input shape:", se_model.input_shape)
    print("Output shape:", se_model.output_shape)
    print("Parameters:", se_model.count_params())

    # Test 3: Residual SE block
    residual_output = residual_se_block(
        inputs,
        filters=64,
        strides=2,
        reduction=4,
        name="test_residual"
    )

    residual_model = tf.keras.Model(
        inputs,
        residual_output
    )

    print("\n3. Residual SE Block")
    print("Input shape:", residual_model.input_shape)
    print("Output shape:", residual_model.output_shape)
    print("Parameters:", residual_model.count_params())

    # Test actual forward pass
    dummy_input = tf.random.normal(
        shape=(2, 64, 64, 32)
    )

    output = residual_model(
        dummy_input,
        training=False
    )

    print("\n4. Forward Pass")
    print("Input batch:", dummy_input.shape)
    print("Output batch:", output.shape)

    assert output.shape == (2, 32, 32, 64)

    print("\nALL BLOCK TESTS PASSED!")