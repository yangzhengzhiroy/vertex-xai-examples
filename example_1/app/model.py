import keras
import tensorflow as tf
from keras import layers as tf_layers

from app.entities import feat_ls


INPUT_SHAPE = (len(feat_ls),)
INPUT_SIGNATURE = {
    f"feat{idx}": tf.TensorSpec(shape=[], dtype=tf.float32, name=f"feat{idx}") 
    for idx in range(1, INPUT_SHAPE[0]+1)
}

def build_model(input_shapes: tuple[int], seed: int):
    initializer = keras.initializers.RandomNormal(mean=0., stddev=1., seed=seed)
    inputs = keras.Input(shape=input_shapes, dtype=tf.float32)
    d_layer = tf_layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(0.01),
        kernel_initializer=initializer,
    )(inputs)
    d_layer = tf_layers.Dropout(0.2)(d_layer)
    d_layer = tf_layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(0.01),
        kernel_initializer=initializer,
    )(d_layer)
    d_layer = tf_layers.Dropout(0.2)(d_layer)
    d_layer = tf_layers.Dense(
        16,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(0.01),
        kernel_initializer=initializer,
    )(d_layer)
    d_layer = tf_layers.Dropout(0.2)(d_layer)
    outputs = tf_layers.Dense(
        1,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(0.01),
        kernel_initializer=initializer,
    )(d_layer)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


class CustomTFModel(tf.Module):

    def __init__(self, seed: int, output_name: str):
        super().__init__()
        self.model = build_model(INPUT_SHAPE, seed)
        self.output_name = output_name

    @tf.function(input_signature=[INPUT_SIGNATURE])
    def __call__(self, feat_dict: dict[str, float]):
        input_tensor = [feat_dict[col] for col in feat_ls]
        input_tensor = tf.convert_to_tensor([input_tensor], dtype=tf.float32)
        resp = {self.output_name: self.model(input_tensor)}
        return resp


if __name__ == "__main__":
    model = CustomTFModel(42, "output")
    # skip training steps...
    model_path = "test_model"
    tf.saved_model.save(
        model,
        model_path,
        signatures={
            "serving_default": model.__call__.get_concrete_function(INPUT_SIGNATURE)
        }
    )
