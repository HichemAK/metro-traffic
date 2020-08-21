import numpy as np
import tensorflow as tf

from metro_traffic.keras_version.attention_rnn import AttentionRNNFuture

model = AttentionRNNFuture(1, Ty=6, hidden_size_encoder=256, hidden_size_decoder=256,
                           dropout=0.1)
model.predict((np.random.rand(1, 6, 116), np.random.rand(1, 6, 115)))
model.load_weights('../../best_model_future.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
tflite_model_name = "mymodel.tflite"
open(tflite_model_name, "wb").write(tflite_model)
