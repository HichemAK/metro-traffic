import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def main():
    df = pd.read_csv('Metro_Interstate_Traffic_Volume.csv')
    df.date_time = pd.to_datetime(df.date_time)

    df = df.drop_duplicates('date_time')
    df.holiday = df.holiday != 'None'
    tf_idf = TfidfVectorizer(stop_words=['is', 'with', 'and'])
    t = tf_idf.fit_transform(df.weather_description)
    df = pd.concat([df, pd.DataFrame(data=t.toarray(), index=df.index).add_prefix('tag_')], axis=1)
    df.drop(columns='weather_description', inplace=True)

    df['hour'] = df.date_time.dt.hour
    df['weekday'] = df.date_time.dt.day_name()
    df['day'] = df.date_time.dt.day
    df['month'] = df.date_time.dt.month_name()
    df.holiday = df.holiday.astype(int)

    df = df.join(pd.get_dummies(df.weather_main, prefix='weather'))
    df = df.join(pd.get_dummies(df.hour, prefix='hour'))
    df = df.join(pd.get_dummies(df.weekday, prefix='weekday'))
    df = df.join(pd.get_dummies(df.day, prefix='day'))
    df = df.join(pd.get_dummies(df.month, prefix='month'))

    df = df.drop(columns=['weather_main', 'hour', 'weekday', 'day', 'month'])
    df = df.sort_values('date_time')

    train = df.loc[df.date_time.dt.year != 2018].copy()
    test = df.loc[df.date_time.dt.year == 2018].copy()

    train.to_csv('train.csv', index=False)
    test.to_csv('test.csv', index=False)
    joblib.dump(tf_idf, '../tf_idf')


if __name__ == '__main__':
    main()
