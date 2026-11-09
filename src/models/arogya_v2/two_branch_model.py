
import tensorflow as tf
from tensorflow.keras import layers, Model, Input

from ablation_models import build_ablation_model


def build_two_branch_model(input_shape=(128, 128, 1)):

    # ==========================================
    # IMAGE BRANCH: Existing A3 architecture
    # ==========================================

    a3_model = build_ablation_model(
        "A3",
        input_shape=input_shape
    )

    # A3 layers: GAP -> Dropout -> Dense
    # Extract feature vector immediately after GAP
    gap_layer = a3_model.layers[-3]

    assert isinstance(
        gap_layer,
        layers.GlobalAveragePooling2D
    ), "Unexpected A3 architecture structure"

    image_encoder = Model(
        inputs=a3_model.input,
        outputs=gap_layer.output,
        name="image_encoder"
    )

    image_input = Input(
        shape=input_shape,
        name="xray_image"
    )

    image_features = image_encoder(image_input)

    # ==========================================
    # METADATA BRANCH
    # 1. Mean intensity
    # 2. Aspect ratio
    # 3. Log image area
    # ==========================================

    metadata_input = Input(
        shape=(3,),
        name="image_metadata"
    )

    metadata_features = layers.Dense(
        16,
        activation="relu",
        name="metadata_dense_1"
    )(metadata_input)

    metadata_features = layers.Dense(
        8,
        activation="relu",
        name="metadata_dense_2"
    )(metadata_features)

    # ==========================================
    # FEATURE FUSION
    # ==========================================

    combined = layers.Concatenate(
        name="feature_fusion"
    )([
        image_features,
        metadata_features
    ])

    combined = layers.Dropout(0.35)(combined)

    output = layers.Dense(
        1,
        activation="sigmoid",
        name="pneumonia_probability"
    )(combined)

    model = Model(
        inputs=[image_input, metadata_input],
        outputs=output,
        name="ArogyaNet_TwoBranch"
    )

    return model


if __name__ == "__main__":

    model = build_two_branch_model()

    print("\nModel:", model.name)
    print("Parameters:", model.count_params())

    dummy_image = tf.random.normal(
        (2, 128, 128, 1)
    )

    dummy_metadata = tf.random.normal(
        (2, 3)
    )

    output = model(
        [dummy_image, dummy_metadata]
    )

    print("Output shape:", output.shape)

    assert output.shape == (2, 1)

    model.summary()

    print("\nTWO-BRANCH ARCHITECTURE TEST PASSED!")