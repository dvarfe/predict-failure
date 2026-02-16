import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Данный класс переопределяется от данных к данным и отвечает за то, чтобы
# привести временную шкалу к формату {0, 1, ...}, также преименовывает столбец со временем
# class TimeTransformer_Old(BaseEstimator, TransformerMixin):
#     def __init__(self, time_column='time', inplace=True, rename=True):
#         '''
#         date_column - Название колонки, которая отвечает за время
#         inplace - Будут ли преобразования происходить inplace
#         rename - Будет ли переименована колонка со временем в time
#         '''
#         self.time_column = time_column
#         self.inplace = inplace
#         self.rename = rename

#     def fit(self, X, y=None):
#         time_df = pd.to_datetime(X[self.time_column])
#         self.min_date = time_df.min()
#         return self

#     def transform(self, X, y=None):
#         # Выбор между изменением на месте или созданием копии
#         X_copy = X if self.inplace else X.copy()

#         # Конвертация 'date' в datetime
#         X_copy[self.time_column] = pd.to_datetime(X_copy[self.time_column])
#         # Приведение временной шкалы
#         X_copy[self.time_column] = X_copy[self.time_column] - self.min_date
#         X_copy[self.time_column] = X_copy[self.time_column].dt.days.astype(int)

#         if self.rename:
#             X_copy = X_copy.rename(columns={self.time_column : 'time'})
#         return X_copy

# Данный класс переопределяется от данных к данным и отвечает за то, чтобы
# привести временную шкалу к формату {0, 1, ...}, также преименовывает столбец со временем и сохраняет только те признаки, которые встречались при обучении
class TimeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, id_col='id', time_column='time', event_column='event', inplace=False, rename=True):
        '''
        date_column - Название колонки, которая отвечает за время
        inplace - Будут ли преобразования происходить inplace
        rename - Будет ли переименована колонка со временем в time
        '''
        self.id_col = id_col
        self.time_column = time_column
        self.event_column = event_column
        self.inplace = inplace
        self.rename = rename

    def fit(self, X, y=None):
        self.features = set(X.columns).difference([self.id_col, self.time_column, self.event_column])
        return self

    def transform(self, X, y=None):
        # Выбор между изменением на месте или созданием копии
        X_copy = X if self.inplace else X.copy()

        # Конвертация 'date' в datetime
        X_copy[self.time_column] = pd.to_datetime(X_copy[self.time_column])
        # Приведение временной шкалы
        X_copy[self.time_column] = X_copy[self.time_column] - \
            X_copy.groupby(self.id_col)[self.time_column].transform('min')
        X_copy[self.time_column] = X_copy[self.time_column].dt.days.astype(int)
        X_copy = X_copy.drop(columns=X_copy.columns[~X_copy.columns.isin(list(self.features.union(
            [self.id_col, self.time_column, self.event_column])))])

        X_copy['max_lifetime'] = X_copy.groupby(self.id_col)[self.time_column].transform('max')
        if self.event_column in X_copy.columns:
            X_copy[self.event_column] = X_copy.groupby(self.time_column)[self.event_column].transform('max')

        if self.rename:
            if self.event_column in X_copy.columns:
                X_copy = X_copy.rename(columns={self.time_column: 'time',
                                                self.id_col: 'serial_number', self.event_column: 'failure'})
            else:
                X_copy = X_copy.rename(columns={self.time_column: 'time',
                                                self.id_col: 'serial_number'})

        return X_copy

# Преобразование, которое переименовывает колонку с событиями в event,
# а колонку с id в id для единообразия, а также удаляет наблюдения с >1 событием


class InitTransforms(BaseEstimator, TransformerMixin):
    def __init__(self, event_column='event', id_column='id', inplace=False):
        '''
        event_column - Название колонки с меткой события
        id_column - Название колонки с идентификатором объекта
        inplace - Будут ли преобразования происходить inplace
        '''
        self.event_column = event_column
        self.id_column = id_column
        self.inplace = inplace

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Выбор между изменением на месте или созданием копии
        X_copy = X if self.inplace else X.copy()

        # Получаем количество событий для дисков
        X_events = X_copy[[self.id_column, self.event_column]].groupby('id').sum()

        # Получаем категории, где событий > 1
        drop_categories = X_events[X_events[self.event_column] > 1].index

        # Удаляем строки с выбранными serial_number
        X_copy = X_copy.drop(X_copy[X_copy[self.id_column].isin(drop_categories)].index).reset_index(drop=True)

        return X_copy

# Класс, который переименовывает столбцы датафрейма в id, event, time
# Использовать необязательно


class ColRenamer(BaseEstimator, TransformerMixin):
    def __init__(self, event_column='failure', id_column='serial_number',
                 time_column='time', inplace=False):
        '''
        event_column - Название колонки с меткой события
        id_column - Название колонки с идентификатором объекта
        inplace - Будут ли преобразования происходить inplace
        '''
        self.event_column = event_column
        self.id_column = id_column
        self.time_column = time_column
        self.inplace = inplace

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Выбор между изменением на месте или созданием копии
        X_copy = X if self.inplace else X.copy()

        X_copy = X_copy.rename(columns={self.event_column: 'event',
                                        self.id_column: 'id',
                                        self.time_column: 'time'})

        return X_copy

# Класс для удаления Truncated наблюдений.


class TruncRemover(BaseEstimator, TransformerMixin):
    def __init__(self, inplace=False, time_col='time', id_col='id', event_col='event'):
        self.inplace = inplace
        self.time_col = time_col
        self.id_col = id_col
        self.event_col = event_col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X if self.inplace else X.copy()
        last_observation = X_copy[self.time_col].max()
        trunc_id = X_copy[(X_copy[self.time_col] == last_observation) &
                          (X_copy[self.event_col] != 1)][self.id_col].unique()
        X_copy.drop(X_copy[X_copy[self.id_col].isin(trunc_id)].index, inplace=True)
        return X_copy

# Класс, который позволяет получать подвыборки по заданной выборке


class Sampler():
    def __init__(self, df, random_state=42, mode='random', time_col='time', id_col='id'):
        '''
        mode in {'random', 'first'}
        '''
        self.time_col = time_col
        self.id_col = id_col

        # Изначально использовалась булева маска last_observ_in_group, но pandas на это выдаёт ворнинги
        # self.last_observ_in_group = df[time_col] == df.groupby(id_col)[time_col].transform('max')

        # self.shuffled_df_idx - GroupBy object, который содержит объекты сгруппированные по id

        if mode == 'random':
            # Оставляем только промежуточные события для каждого объекта(не первое и не последнее)
            self.shuffled_df_idx = (df[~(df[self.time_col] == df.groupby(self.id_col)[self.time_col].transform('max')) &
                                       ~(df[self.time_col] == df.groupby(self.id_col)[self.time_col].transform('min'))]
                                    .sample(frac=1, random_state=random_state).
                                    # Оставляем только колонку с id и группируем(нам понадобятся только индексы, они задают фиксированную перестановку налюдений в цепочке)
                                    loc[:, [self.id_col]].
                                    groupby(self.id_col))
        elif mode == 'first':
            df = df.sort_values(by=self.time_col)
            self.shuffled_df_idx = (df[~(df[self.time_col] == df.groupby(self.id_col)[self.time_col].transform('max')) &
                                       ~(df[self.time_col] == df.groupby(self.id_col)[self.time_col].transform('min'))]
                                    .loc[:, [self.id_col]]
                                    .groupby(self.id_col))
        self.X = df

    def get_n_samples(self, n_samples=10):
        # Получаем n промежуточных наблюдений
        df_sampled = self.X.loc[self.shuffled_df_idx.head(n_samples - 1).index, :]
        # Последняя запись в каждой цепочке наблюдений должна отражать последнее состояние диска
        df_sampled = pd.concat([df_sampled,
                                self.X[
                                    (self.X[self.time_col] == self.X.groupby(self.id_col)[self.time_col].transform('max')) |
                                    (self.X[self.time_col] == self.X.groupby(
                                        self.id_col)[self.time_col].transform('min'))
                                ]])  # Добавляем к промежуточным наблюдениям первое и последнее наблюдения
        return df_sampled


# Класс, предназначенный для удаления признаков с большим числом пропусков,
# либо признаков, заданных в features_to_remove.
class FeatureFilter(BaseEstimator, TransformerMixin):
    def __init__(self, nan_fraction=0.5, features_to_remove=None, event_col='failure', inplace=False):
        '''
        nan_fraction - допустимая доля пропусков
        features_to_remove - список названий признаков, которые необходимо обязательно удалить
        '''
        self.event_col = event_col
        self.features_to_remove = features_to_remove
        self.nan_fraction = nan_fraction
        self.inplace = inplace

    def fit(self, X, y=None):
        na_count = X.isna().sum()
        high_na = [
            c for c in X.columns
            if na_count[c] > X.shape[0] * self.nan_fraction
        ]
        self.features_to_remove_ = set((self.features_to_remove or []) + high_na + [self.event_col])
        # print(X.columns)
        self.features = list(set(X.columns).difference(self.features_to_remove_))
        # print(self.features)
        return self

    def transform(self, X, y=None):
        # Выбор между изменением на месте или созданием копии
        X_copy = X if self.inplace else X.copy()
        print(self.event_col in X_copy.columns)
        features = (self.features + [self.event_col]) if self.event_col in X_copy.columns else self.features
        return X_copy[features]

# Класс, который объединяет наблюдения, которые произошли в один момент времени


class ObservationAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, id_col='id', time_col='time', inplace=False):
        self.inplace = inplace
        self.id_col = id_col
        self.time_col = time_col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Выбор между изменением на месте или созданием копии
        X_copy = X if self.inplace else X.copy()

        X_copy = X_copy.drop_duplicates(subset=[self.id_col, self.time_col], keep='last')

        return X_copy

# Класс для заполнения пропусков в данных. Скорее всего в тестовой выборке заполнять будем константой.


class NanImputer(BaseEstimator, TransformerMixin):
    def __init__(self, fill_val=0, id_col='id', time_col='time', inplace=False):
        '''
        fill_val - значение для заполнения пропусков по умолчанию.
                  Используется, если самое первое наблюдение содержит nan.
        '''
        self.inplace = inplace
        self.fill_val = fill_val
        self.id_col = id_col
        self.time_col = time_col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Выбор между изменением на месте или созданием копии
        X_copy = X if self.inplace else X.copy()

        # Сортировка, чтобы bfill работал корректно
        X_copy = X_copy.sort_values(by=[self.id_col, self.time_col])

        # Применяем bfill для каждого диска
        X_copy.loc[:, X_copy.columns != self.id_col] = (X_copy
                                                        .groupby(self.id_col)
                                                        .transform('bfill')
                                                        .infer_objects(copy=False)
                                                        .fillna(self.fill_val)
                                                        )

        return X_copy

# Функция для разделения выборки на тренировочную и валидационную


def stratified_split(df, test_size=0.2, random_state=42, id_col='id', event_col='event'):
    id_events = df.groupby(id_col)[event_col].max().reset_index()
    train_id, test_id = train_test_split(id_events[id_col], test_size=test_size, random_state=random_state,
                                         stratify=id_events[event_col])

    df_train = df[df[id_col].isin(train_id)]
    df_test = df[df[id_col].isin(test_id)]

    return df_train, df_test


class TimeMaskedScaler(BaseEstimator, TransformerMixin):
    def __init__(self, features, id_col='id', time_col='time'):
        self.features = features
        self.id_col = id_col
        self.time_col = time_col
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        X = X.copy()
        mask = (
            X.groupby(self.id_col)[self.time_col].transform('max')
            != X[self.time_col]
        )
        self.features_ = list(set(self.features).intersection(X.columns))
        self.scaler.fit(X.loc[mask, self.features_])
        return self

    def transform(self, X):
        X = X.copy()
        X.loc[:, self.features_] = self.scaler.transform(X[self.features_])
        return X
