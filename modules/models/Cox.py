import pandas as pd
import numpy as np
import torch
from lifelines import CoxTimeVaryingFitter
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional
from ..base.model_base import AbstractModel


class CoxTimeVaryingEstimator(AbstractModel):
    """
    CoxTimeVaryingEstimator: полностью совместим с новой инфраструктурой.
    Формирует корректные интервалы start/stop для TV-Cox на основе time и duration.
    """

    def __init__(self, penalizer=0.0, l1_ratio=0.0, event_col="failure", time_col='time', id_col=None, device=None):
        self.model = CoxTimeVaryingFitter(penalizer=penalizer, l1_ratio=l1_ratio)
        self.event_col = event_col
        self.time_col = time_col
        self.id_col = id_col or 'id'
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_cols = None
        self.is_fitted = False

    def _batch_to_df(self, batch):
        """Конвертирует батч из DiskDataset формата в DataFrame"""
        # DiskDataset возвращает 5 тензоров размера batch_size
        serial_numbers, times, features_tensor, failures, times_to_event = batch

        serial_numbers = list(serial_numbers)
        times = times.numpy()
        features_tensor = features_tensor.numpy()
        failures = failures.numpy()
        times_to_event = times_to_event.numpy()

        # Создаем словарь со списками для каждой колонки
        data = {
            self.id_col: serial_numbers,
            self.time_col: times.tolist(),
            self.event_col: failures.tolist()
        }

        # Добавляем признаки как отдельные колонки
        num_features = features_tensor.shape[1]
        for j in range(num_features):
            data[f'feature_{j}'] = features_tensor[:, j].tolist()

        return pd.DataFrame(data)

    def fit(self, train_dataloader: DataLoader, id_col: Optional[str] = None):
        """Обучить модель на DataLoader"""
        dfs = []
        for batch in tqdm(train_dataloader, desc="Collecting data for Cox fit"):
            batch_df = self._batch_to_df(batch)
            dfs.append(batch_df)
        df_all = pd.concat(dfs, ignore_index=True)
        return self.fit_dataframe(df_all, id_col=id_col)

    def fit_dataframe(self, df_all: pd.DataFrame, id_col: Optional[str] = None):
        df_tv = self._to_start_stop(df_all)

        try:
            self.model.fit(df_tv, id_col=self.id_col, start_col='start', stop_col='stop', event_col=self.event_col)
            self.is_fitted = True
            self.feature_cols = [c for c in df_tv.columns if c not in [
                self.id_col, 'start', 'stop', self.event_col, 'duration']]
            return self
        except (Exception) as e:
            self.is_fitted = False
            print(f"Обучение модели не удалось: {e}")
            return None

    def predict_dataframe(self, data: pd.DataFrame, times=None, id_col: Optional[str] = None) -> pd.DataFrame:
        if times is None:
            times = np.arange(1, 11)

        n_samples = len(data)

        available_features = [c for c in self.feature_cols if c in data.columns]
        if not available_features:
            raise ValueError("No features available")

        X_features = data[available_features]
        survival_probs = self._get_survival_function(X_features, times)

        # Создаем DataFrame с результатами
        cols = [f"{t}" for t in times]
        df = pd.DataFrame(data=survival_probs, index=range(n_samples), columns=cols)

        if id_col is not None and id_col in data.columns:
            df['id'] = data[id_col].astype(str).values
        else:
            df['id'] = data.index.astype(str).values

        return df

    def fit_dataloader(self, train_dataloader: DataLoader):
        dfs = []
        for batch in tqdm(train_dataloader, desc="Collecting data for Cox fit"):
            batch_df = self._batch_to_df(batch)
            dfs.append(batch_df)
        df_all = pd.concat(dfs, ignore_index=True)
        return self.fit_dataframe(df_all)

    def _to_start_stop(self, df):
        """
        Формирует DataFrame для CoxTimeVaryingFitter:
        - start = time - min_time_in_group
        - stop = start следующего наблюдения, для последнего: start + 1
        - event_col для всех кроме последнего: 0 (цензурирование), для последнего — текущее значение
        """
        df = df.sort_values([self.id_col, self.time_col]).copy()
        min_times = df.groupby(self.id_col)[self.time_col].transform('min')
        df['start'] = df[self.time_col] - min_times
        df['stop'] = df.groupby(self.id_col)['start'].shift(-1)
        last_mask = df['stop'].isna()
        df.loc[last_mask, 'stop'] = df.loc[last_mask, 'start'] + \
            1  # Добавляем единицу времени для последнего наблюдения
        df.loc[~last_mask, self.event_col] = 0  # Все наблюдения, кроме последнего цензурированы
        return df

    def _get_survival_function(self, X_features, times):
        baseline_surv = self.model.baseline_survival_
        fill_indices = baseline_surv.index.searchsorted(times, side='right') - 1
        fill_indices = np.clip(fill_indices, 0, max(fill_indices) - 1)
        baseline_surv_interp = baseline_surv.iloc[fill_indices, 0].values
        partial_haz = self.model.predict_partial_hazard(X_features).values
        surv = baseline_surv_interp[None, :] ** partial_haz[:, None]
        return surv

    def predict(self, dataloader: DataLoader, times: np.ndarray, id_col: Optional[str] = None):
        """
        Собирает все данные из DataLoader в один DataFrame, затем предсказывает survival-функции для всех сразу.
        Возвращает: (df_surv, df_gt)
        """
        # Сначала соберём весь датасет в один DataFrame
        dfs = []
        for batch in tqdm(dataloader, desc="Collecting data for Cox prediction"):
            batch_df = self._batch_to_df(batch)
            dfs.append(batch_df)
        df_all = pd.concat(dfs, ignore_index=True)

        X_feat = df_all[self.feature_cols]
        times = np.array(times)
        surv = self._get_survival_function(X_feat, times)
        pred_values = np.column_stack([df_all[self.time_col].values, surv])
        serial_numbers_flat = df_all[self.id_col].values
        columns = ['time'] + times.tolist()
        df_surv = pd.DataFrame(pred_values, columns=columns)
        ids = serial_numbers_flat.astype(str)
        df_surv.insert(0, 'serial_number', ids)
        df_surv.index = ids
        df_surv['time'] = df_surv['time'].astype('int32')

        # gt (ground truth)
        if 'duration' in df_all:
            gt_values = np.column_stack([
                df_all[self.time_col].values,
                df_all['duration'].values,
                df_all[self.event_col].values
            ])
            df_gt = pd.DataFrame(gt_values, columns=['time', 'duration', 'failure'])
            df_gt.insert(0, 'serial_number', serial_numbers_flat)
            df_gt = df_gt.astype({'serial_number': 'string', 'time': 'int32', 'duration': 'int32'})
            df_gt['failure'] = df_gt['failure'] == 1
        else:
            df_gt = pd.DataFrame()

        return df_surv

    def get_expected_time(self, dataloader: DataLoader, times: np.ndarray, id_col: Optional[str] = None):
        df_surv, df_gt = self.predict(dataloader, times, id_col=id_col)
        return self.get_expected_time_by_predictions(df_surv, times), df_gt

    def get_expected_time_by_predictions(self, X_pred: pd.DataFrame, times: np.ndarray):
        survival_vec = X_pred.drop(['serial_number', 'time'], axis='columns').values
        return np.trapezoid(y=survival_vec, x=times)
