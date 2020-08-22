import joblib
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from numpy.random import seed
from sklearn.model_selection import train_test_split

from metro_traffic.keras_version.attention_rnn import AttentionRNNFuture
from metro_traffic.utils import CustomStandardScaler, shuffle_jointly

tf.random.set_seed(42)
seed(42)

train = pd.read_csv('../train.csv')
train.drop(columns='date_time', inplace=True)

packet_size = 6
x1s = train.values[:-packet_size]
x2s = train.drop(columns='traffic_volume').values[packet_size:]
targets = train['traffic_volume'].values[packet_size:]

x1s_train, x1s_test, x2s_train, x2s_test, target_train, target_test = train_test_split(x1s, x2s, targets,
                                                                                       shuffle=False, test_size=0.1)
ss = CustomStandardScaler(num=3)
target_train = target_train[:, np.newaxis]
target_test = target_test[:, np.newaxis]
x1s_train, x2s_train, target_train = ss.fit_transform([x1s_train, x2s_train, target_train])
x1s_test, x2s_test, target_test = ss.transform([x1s_test, x2s_test, target_test])

target_train, target_test = np.expand_dims(target_train, -1), np.expand_dims(target_test, -1)


print(ss.ss[-1].mean_, np.sqrt(ss.ss[-1].var_))

data_train = [x1s_train, x2s_train]
data_test = [x1s_test, x2s_test]


def generator(data_, target_, stride, step, batch_size):
    while True:
        inputs = [0] * len(data_)
        for t in range(0, stride, step):
            data = [0] * len(data_)
            for i in range(len(data)):
                data[i] = data_[i][t:t + stride * ((data_[i].shape[0] - t) // stride)]
                data[i] = data[i].reshape(data[i].shape[0] // stride, stride, data[i].shape[-1])
            target = target_[t:t + stride * ((target_.shape[0] - t) // stride)]
            target = target.reshape(target.shape[0] // stride, stride, target.shape[-1])
            data.append(target)
            data = shuffle_jointly(*data)
            target = data.pop(-1)
            for i in range(0, data[0].shape[0], batch_size):
                for j in range(len(data)):
                    inputs[j] = data[j][i:i + batch_size]

                y = target[i:i + batch_size]
                yield inputs, y


step = 2

batch_size = 32
# GridSearch
path = 'grid_search_results.csv'
dropouts = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
learning_rates = [1e-3, 3e-3]
hidden_sizes = [512, 256, 128, 64, 32]

df = pd.DataFrame(columns=['dropout', 'lr', 'hs', 'loss', 'val_loss'])

for dropout in dropouts:
    for lr in learning_rates:
        for hs in hidden_sizes:
            tf.random.set_seed(42)
            seed(42)
            print("Begin Training...", (dropout, lr, hs))
            model = AttentionRNNFuture(1, Ty=packet_size, hidden_size_encoder=hs, hidden_size_decoder=hs,
                                       dropout=dropout)
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                          loss=keras.losses.MeanSquaredError())
            history = model.fit(generator(data_train, target_train, packet_size, step, batch_size=32), epochs=20,
                                validation_data=generator(data_test, target_test, packet_size, step,
                                                          batch_size=batch_size),
                                steps_per_epoch=data_train[0].shape[0] // (batch_size * packet_size),
                                validation_steps=data_test[0].shape[0] // (batch_size * packet_size))
            df2 = pd.DataFrame(data=history.history)
            df2['dropout'] = dropout
            df2['lr'] = lr
            df2['hs'] = hs
            df = pd.concat([df, df2], axis=0)
            df.to_csv(path, index=False)
