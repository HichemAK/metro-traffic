import tensorflow as tf

def rmse_per_pos(y_true, y_pred):
    acc = tf.squeeze(tf.reduce_mean(tf.sqrt((y_true - y_pred)**2), axis=0), axis=-1)
    return acc