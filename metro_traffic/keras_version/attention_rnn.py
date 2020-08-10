import tensorflow as tf
from keras import Model
from keras import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Lambda, Concatenate, Softmax


class AttentionRNN(Model):
    def __init__(self, input_size, output_size, Ty, hidden_size_encoder=128,
                 hidden_size_decoder=128, dropout=0.1):
        super(AttentionRNN, self).__init__()
        self.Ty = Ty
        self.dropout = dropout
        # Encoder
        self.bi_lstm = Bidirectional(
            LSTM(units=hidden_size_encoder, dropout=dropout, recurrent_dropout=dropout, return_sequences=True))

        # Decoder
        self.lstm = LSTM(units=hidden_size_decoder, dropout=dropout, recurrent_dropout=dropout, return_state=True)
        self.attention = Sequential([Dense(80, activation='relu'), Dense(1)])
        self.dense = Dense(output_size)
        self.concatenate = Concatenate(axis=-1)
        self.squeeze = Lambda(lambda x : tf.squeeze(x))
        self.expand_dims = Lambda(lambda x : tf.expand_dims(x, axis=1))
        self.repeat = Lambda(lambda var : tf.repeat(var[0], var[1], axis=1))
        self.softmax = Softmax()
        self.reduce_sum = Lambda(lambda var : tf.reduce_sum(var[0] * tf.expand_dims(var[1], -1), axis=1))
        self.stack = Lambda(lambda x : tf.stack(x))

    def call(self, x1):
        if isinstance(x1, tuple):
            x1 = x1[0]
        x1 = self.bi_lstm(x1)
        state = (tf.zeros((x1.shape[0], self.lstm.units)), tf.zeros((x1.shape[0], self.lstm.units)))
        outputs = []
        for _ in range(self.Ty):
            repeat = self.expand_dims(state[0])
            repeat = self.repeat((repeat, x1.shape[1]))
            repeat = self.concatenate([x1, repeat])
            attention = self.attention(repeat)
            attention = self.squeeze(attention)
            attention = self.softmax(attention)
            attention = self.reduce_sum((x1, attention))
            attention = self.expand_dims(attention)
            res, state1, state2 = self.lstm(attention, initial_state=state)
            state = state1, state2
            outputs.append(res)
        outputs = self.stack(outputs)
        return self.dense(outputs)
