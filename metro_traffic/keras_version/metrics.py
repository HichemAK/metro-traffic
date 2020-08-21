import tensorflow as tf


def rmse_per_pos(y_true, y_pred):
    acc = tf.squeeze(tf.reduce_mean(tf.sqrt((y_true - y_pred) ** 2), axis=0), axis=-1)
    return acc


def mse_per_pos_single(i):
    def _mse_per_pos_single(y_true, y_pred):
        y_true = y_true[:, i]
        y_pred = y_pred[:, i]
        return tf.reduce_mean((y_true - y_pred) ** 2)

    _mse_per_pos_single.__name__ += '_' + str(i)
    return _mse_per_pos_single
