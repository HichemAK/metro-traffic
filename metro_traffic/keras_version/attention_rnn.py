import keras
import tensorflow as tf
from keras import Model
from keras import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Concatenate, Lambda

class AttentionRNNFuture(Model):
    def __init__(self, output_size, Ty, hidden_size_encoder=128,
                 hidden_size_decoder=128, dropout=0.1):
        super(AttentionRNNFuture, self).__init__()
        self.Ty = Ty
        self.dropout = dropout
        # Encoder
        self.bi_lstm = Bidirectional(
            LSTM(units=hidden_size_encoder, dropout=dropout, return_sequences=True))
        self.bi_lstm2 = Bidirectional(
            LSTM(units=hidden_size_encoder, dropout=dropout, return_sequences=True))

        # Decoder
        self.decoder = AttentionDecoder(Ty, hidden_size_decoder, dropout)
        self.dense = Dense(output_size)

    def call(self, x):
        if isinstance(x, tuple):
            x1, x2 = x
        x1 = self.bi_lstm(x1)
        x2 = self.bi_lstm2(x2)
        x1 = tf.concat([x1, x2], axis=1)
        outputs = self.decoder(x1)
        return self.dense(outputs)


class AttentionRNN(Model):
    def __init__(self, output_size, Ty, hidden_size_encoder=128,
                 hidden_size_decoder=128, dropout=0.1):
        super(AttentionRNN, self).__init__()
        self.Ty = Ty
        self.dropout = dropout
        # Encoder
        self.bi_lstm = Bidirectional(
            LSTM(units=hidden_size_encoder, dropout=dropout, return_sequences=True))

        # Decoder
        self.decoder = AttentionDecoder(Ty, hidden_size_decoder, dropout)
        self.dense = Dense(output_size)

    def call(self, x1):
        if isinstance(x1, tuple):
            x1 = x1[0]
        x1 = self.bi_lstm(x1)
        outputs = self.decoder(x1)
        return self.dense(outputs)

class AttentionDecoder(keras.layers.Layer):
    def __init__(self, Ty, hidden_size_decoder, dropout):
        super().__init__()
        self.lstm = LSTM(units=hidden_size_decoder, dropout=dropout, return_state=True)
        self.attention = Sequential([Dense(80, activation='relu'), Dense(1)])
        self.concatenate = Concatenate(axis=-1)
        self.Ty = Ty
        self.initial_state = Lambda(lambda var : (tf.zeros((tf.shape(var[0])[0], var[1])), ) * 2)

    def call(self, x):
        state = self.initial_state((x, self.lstm.units))
        outputs = []
        for _ in range(self.Ty):
            repeat = tf.expand_dims(state[0], axis=1)
            repeat = tf.repeat(repeat, tf.shape(x)[1], axis=1)
            repeat = self.concatenate([x, repeat])
            attention = self.attention(repeat)
            attention = tf.nn.softmax(attention, axis=-2)
            attention = tf.reduce_sum(x * attention, axis=1)
            attention = tf.expand_dims(attention, axis=1)
            res, state1, state2 = self.lstm(attention, initial_state=state)
            state = state1, state2
            outputs.append(res)
        outputs = tf.stack(outputs, axis=1)
        return outputs
