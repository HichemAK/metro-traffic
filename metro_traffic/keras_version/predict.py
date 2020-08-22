import keras
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from metro_traffic.utils import CustomStandardScaler


def predict(model: keras.Model, standard_scaler: CustomStandardScaler, tf_idf: TfidfVectorizer, df_past, df_future):
    """model : Keras Model"""
    df_past.date_time = pd.to_datetime(df_past.date_time)
    df_future.date_time = pd.to_datetime(df_future.date_time)

    df_past.holiday = df_past.holiday != 'None'
    df_future.holiday = df_future.holiday != 'None'
    t = tf_idf.transform(df_past.weather_description)
    t2 = tf_idf.transform(df_future.weather_description)
    df_past = pd.concat([df_past, pd.DataFrame(data=t.toarray(), index=df_past.index).add_prefix('tag_')], axis=1)
    df_future = pd.concat([df_future, pd.DataFrame(data=t2.toarray(), index=df_future.index).add_prefix('tag_')],
                          axis=1)
    df_past.drop(columns='weather_description', inplace=True)
    df_future.drop(columns='weather_description', inplace=True)

    df_past['hour'] = df_past.date_time.dt.hour
    df_past['weekday'] = df_past.date_time.dt.day_name()
    df_past['day'] = df_past.date_time.dt.day
    df_past['month'] = df_past.date_time.dt.month_name()
    df_past.holiday = df_past.holiday.astype(int)

    df_future['hour'] = df_future.date_time.dt.hour
    df_future['weekday'] = df_future.date_time.dt.day_name()
    df_future['day'] = df_future.date_time.dt.day
    df_future['month'] = df_future.date_time.dt.month_name()
    df_future.holiday = df_future.holiday.astype(int)

    df_past = df_past.join(pd.get_dummies(df_past.weather_main, prefix='weather'))
    df_past = df_past.join(pd.get_dummies(df_past.hour, prefix='hour'))
    df_past = df_past.join(pd.get_dummies(df_past.weekday, prefix='weekday'))
    df_past = df_past.join(pd.get_dummies(df_past.day, prefix='day'))
    df_past = df_past.join(pd.get_dummies(df_past.month, prefix='month'))

    df_future = df_future.join(pd.get_dummies(df_future.weather_main, prefix='weather'))
    df_future = df_future.join(pd.get_dummies(df_future.hour, prefix='hour'))
    df_future = df_future.join(pd.get_dummies(df_future.weekday, prefix='weekday'))
    df_future = df_future.join(pd.get_dummies(df_future.day, prefix='day'))
    df_future = df_future.join(pd.get_dummies(df_future.month, prefix='month'))

    df_past = df_past.drop(columns=['weather_main', 'hour', 'weekday', 'day', 'month'])
    df_future = df_future.drop(columns=['weather_main', 'hour', 'weekday', 'day', 'month'])

    df_past.drop(columns='date_time', inplace=True)
    df_future.drop(columns='date_time', inplace=True)

    traffic = df_past['traffic_volume'].values.reshape(-1, 1)
    df_past.drop(columns='traffic_volume', inplace=True)

    df_past, df_future, traffic = standard_scaler.transform([df_past, df_future, traffic])

    df_future = df_future[np.newaxis, :]
    df_future = df_future[np.newaxis, :]

    y = model.predict((df_past, df_future))
    return standard_scaler.ss[2].inverse_transform(y)
