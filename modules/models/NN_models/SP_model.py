from typing import Optional, Set, Tuple, List
import time
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from tqdm import tqdm

from .ClassifierArchitecture import ClassifierArchitecture
from ...metrics.metrics import get_metrics
from .utils import ibs_remain_torch, ibs_like_loss, masked_loss
from ...base.model_base import AbstractModel

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
np.random.seed(42)

TIMES = np.arange(1, 10)
DEFAULT_HIDDEN_DIM = 64
DEFAULT_LR = 1e-3
DEFAULT_EPOCHS = 100
DEFAULT_DEVICE = None


class SurvPredictor():
    """Survival predictor.

    Attributes:
        device (str): Device where the model and tensors are allocated ('cuda' or 'cpu').
        _model (nn.Module): Neural network model instance.
        criterion (nn.Module): Loss function used for training.
        optimizer (torch.optim.Optimizer): Optimizer for training the model.
        writer (Optional[SummaryWriter]): TensorBoard writer instance.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = DEFAULT_HIDDEN_DIM,
                 lr: float = DEFAULT_LR,
                 epochs: int = DEFAULT_EPOCHS,
                 device: Optional[str] = DEFAULT_DEVICE):
        """Initialize survival predictor.

        Args:
            input_dim: Number of input features
            hidden_dim: Number of hidden units (default: 64)
            lr: Learning rate (default: 1e-3)
            epochs: Number of training epochs (default: 100)
            device: Computation device ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = nn.DataParallel(ClassifierArchitecture(input_dim, hidden_dim).to(self.device))
        self.epochs = epochs
        self.criterion = ibs_like_loss
        self.optimizer = torch.optim.AdamW(self._model.parameters(), lr=lr)

        # Training metrics
        self.loss: List[float] = []
        self.fit_times: List[float] = []
        self.best_loss = float('inf')

    def get_survival_function(self, times, event_times, status):
        surv = torch.ones((len(event_times), len(times)))
        for i, (t, s) in enumerate(zip(event_times, status)):
            surv[i, times >= t] = 0 if s else 1
        return surv

    def fit(self,
            train_dataloader: DataLoader,
            times: np.ndarray,
            val_dataloader: Optional[DataLoader] = None,
            early_stopping: bool = False,
            score_metric: str = 'neg_ci',
            additional_val_metrics: Set[str] = {'ci'},
            patience: int = 10,
            min_delta=0.05,
            val_times: List[int] = None,
            writer: Optional[SummaryWriter] = None):
        """Train model.

        Args:
            train_dataloader: DataLoader providing training batches
            times: Array of time points for survival prediction
            val_dataloader: DataLoader providing validation batches. If None, no validation is performed.
            early_stopping: Whether to use early stopping based on validation performance
            score_metric: Metric to use for evaluating model performance. Options: 'neg_ci' (negative concordance index),
                'ibs' (integrated Brier score), or 'loss' (training loss).
            patience: Number of epochs to wait for improvement before early stopping
            min_delta: Minimum change in the monitored metric to qualify as an improvement
            writer: Optional SummaryWriter for logging training metrics (e.g., for TensorBoard)
    """
        self._model.train()
        times_tensor = torch.as_tensor(times, device=self.device, dtype=torch.float32)
        global_step = 0

        if early_stopping:
            if val_dataloader is None:
                raise ValueError("Validation DataLoader must be provided for early stopping")
            self.best_score = np.inf
            self.early_stop_counter = 0
            self.best_model_state = copy.deepcopy(self._model.state_dict())

        for epoch in range(self.epochs):
            epoch_loss = 0
            epoch_steps = 0
            start_time = time.time()

            with tqdm(train_dataloader, unit='batch') as tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                for _, _, X, y, time_to_event in tepoch:
                    X = X.to(self.device).float()
                    # Compute df_test for ibs
                    survival_test = pd.DataFrame()
                    survival_test['event'] = y
                    survival_test['duration'] = time_to_event

                    y = y.to(self.device).float()
                    y_true_surv = self.get_survival_function(times_tensor, time_to_event, y)
                    y_true_surv = y_true_surv.to(self.device).float()

                    # Forward pass
                    batch_size = X.size(0)
                    expanded_X = X.unsqueeze(1).expand(-1, len(times), -1)
                    expanded_times = times_tensor.view(1, -1, 1).expand(batch_size, -1, -1)

                    hazards = self._model(
                        expanded_X.reshape(-1, expanded_X.size(-1)),
                        expanded_times.reshape(-1, 1)
                    ).view(batch_size, -1)

                    surv_pred = torch.exp(-hazards.cumsum(dim=1))

                    # loss = ibs_remain_torch(None, survival_test.to_records(index=False), surv_pred, times=times_tensor)
                    loss = masked_loss(self.criterion, surv_pred, y_true_surv,
                                       events=y, durations=time_to_event, times=times_tensor)
                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    loss_value = loss.item()
                    epoch_loss += loss_value
                    epoch_steps += 1
                    global_step += 1

                    if writer is not None:
                        writer.add_scalar('Loss/train_batch', loss_value, global_step)

                    tepoch.set_postfix(loss=loss_value)

            # Epoch statistics
            avg_epoch_loss = epoch_loss / epoch_steps
            epoch_time = time.time() - start_time
            self.loss.append(avg_epoch_loss)
            self.fit_times.append(epoch_time)

            if val_dataloader is not None:
                if val_times is None:
                    val_times = times
                val_times_tensor = torch.as_tensor(val_times, device=self.device, dtype=torch.float32)
                with torch.no_grad():
                    X_pred, X_gt = self.predict(val_dataloader, val_times)
                    calc_metrics = additional_val_metrics.union({score_metric.replace('neg_', '')})
                    metrics = get_metrics(self, X_pred, X_gt, val_times, calc_metrics)
                    event_times_val = X_gt['time'] + X_gt['duration']
                    events_val = X_gt['failure']

                    event_times_val_tensor = torch.as_tensor(
                        event_times_val, device=self.device, dtype=torch.float).unsqueeze(1)
                    events_val_tensor = torch.as_tensor(events_val, device=self.device, dtype=torch.float)

                    y_true_surv_val = self.get_survival_function(
                        val_times_tensor, event_times_val_tensor, events_val_tensor)
                    y_true_surv_val = y_true_surv_val.to(self.device).float()

                    survival_test_val = pd.DataFrame()
                    survival_test_val['event'] = X_gt['failure']
                    survival_test_val['duration'] = X_gt['duration']
                    surv_val_pred = torch.Tensor(X_pred.iloc[0:, 2:].astype('float').values)
                    surv_val_pred = surv_val_pred.to(self.device).float()

                    # val_loss = ibs_remain_torch(None, survival_test_val.to_records(index=False),
                    #                             surv_val_pred, times=val_times_tensor)
                    val_loss = masked_loss(self.criterion, surv_val_pred, y_true_surv_val,
                                           events=events_val_tensor, durations=event_times_val_tensor,
                                           times=val_times_tensor).item()

                del X_pred, X_gt, event_times_val_tensor, events_val_tensor, surv_val_pred

                val_metrics = {metric: metrics[metric] for metric in calc_metrics}

                val_score = val_metrics[score_metric] if not score_metric.startswith(
                    'neg_') else -val_score[score_metric]
                print("Val score:", val_score)
                if writer is not None:
                    for metric in val_metrics:
                        writer.add_scalar(f'Val/{metric}', metrics[metric], epoch)
                    writer.add_scalar(f'Val/loss', val_loss, epoch)

                if early_stopping:
                    if val_score < self.best_score - min_delta:
                        self.best_score = val_score
                        self.early_stop_counter = 0
                        self.best_model_state = copy.deepcopy(self._model.state_dict())
                    else:
                        self.early_stop_counter += 1
                        if self.early_stop_counter >= patience:
                            print(f"Early stopping triggered at epoch {epoch}!")
                            self._model.load_state_dict(self.best_model_state)
                            break

            # TensorBoard logging
            if writer is not None:
                writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch)
                writer.add_scalar('Time/epoch', epoch_time, epoch)

    def predict(self, dataloader: DataLoader | pd.DataFrame,
                times: Optional[np.ndarray] = TIMES,
                id_col: Optional[str] = None,
                time_col: str = 'time',
                failure_col: str = 'failure',
                duration_col: str = 'duration',
                max_lifetime_col: str = 'max_lifetime') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Predicts survival functions for observations from the dataloader.

         For each observation, this method predicts the survival probability
         over a predefined set of time points using the trained model.

         Args:
             dataloader (DataLoader): A DataLoader providing batches of data in the form:
                 (serial_numbers, obs_times, X, y, real_durations)
             times (np.ndarray, optional): Array of time points at which the survival function is evaluated.
                 Defaults to TIMES.

         Returns:
             Tuple[pd.DataFrame, pd.DataFrame]:
                 - A DataFrame containing predicted survival functions for each observation.
                   Columns: ['serial_number', 'time', t1, t2, ..., tN]
                 - A DataFrame containing ground truth durations and event indicators if available,
                   otherwise an empty DataFrame.
         """
        model_state = self._model.training
        self._model.eval()
        pred_chunks = []
        pred_serials = []
        gt_chunks = []

        # If a DataFrame is passed (from SystemManager), convert it to an in-memory Dataset
        created_loader = False
        if isinstance(dataloader, pd.DataFrame):
            df = dataloader.copy()

            if id_col is not None and id_col in df.columns:
                id_col_local = id_col
            elif 'serial_number' in df.columns:
                id_col_local = 'serial_number'
            else:
                id_col_local = None

            if duration_col in df.columns:
                duration_series = df[duration_col]
            elif max_lifetime_col in df.columns and time_col in df.columns:
                duration_series = df[max_lifetime_col] - df[time_col]
            else:
                duration_series = pd.Series(-1, index=df.index)

            if failure_col in df.columns:
                failure_series = df[failure_col]
            elif 'event' in df.columns:
                failure_series = df['event']
            else:
                failure_series = pd.Series(0, index=df.index)

            if id_col_local is not None and id_col_local in df.columns:
                id_series = df[id_col_local].astype(str)
            else:
                id_series = df.index.astype(str)

            # Feature columns: all numeric columns except id/time/event/duration
            exclude = {id_col_local, time_col, failure_col, 'event', duration_col, max_lifetime_col}
            feature_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]

            class DataFrameDataset(torch.utils.data.Dataset):
                def __init__(self, df, id_series, time_col, feature_cols, duration_series, failure_series):
                    self.df = df
                    self.id_series = id_series.reset_index(drop=True)
                    self.time_col = time_col
                    self.feature_cols = feature_cols
                    self.duration_series = duration_series.reset_index(drop=True)
                    self.failure_series = failure_series.reset_index(drop=True)

                def __len__(self):
                    return len(self.df)

                def __getitem__(self, idx):
                    row = self.df.iloc[idx]
                    serial = str(self.id_series.iloc[idx])
                    t = int(row[self.time_col]) if self.time_col in self.df.columns and not pd.isna(
                        row[self.time_col]) else 0
                    if self.feature_cols:
                        X = torch.tensor(row[self.feature_cols].astype(float).values, dtype=torch.float32)
                    else:
                        X = torch.tensor([], dtype=torch.float32)
                    y = int(self.failure_series.iloc[idx])
                    duration = int(self.duration_series.iloc[idx]) if not pd.isna(
                        self.duration_series.iloc[idx]) else -1
                    return serial, t, X, y, duration

            dataset = DataFrameDataset(df, id_series, time_col, feature_cols, duration_series, failure_series)
            dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
            created_loader = True

        with torch.no_grad():
            times = np.asarray(times)
            times_tensor = torch.as_tensor(times, device=self.device, dtype=torch.float32)
            n_times = len(times)

            for serial_numbers, obs_times, X, y, real_durations in tqdm(dataloader):
                batch_size = X.size(0)
                serial_numbers = np.array(serial_numbers)

                X = X.to(self.device)
                obs_times = obs_times.to(self.device).int()

                expanded_X = X.unsqueeze(1).expand(-1, n_times, -1)
                expanded_times = times_tensor.reshape(1, -1, 1).expand(batch_size, -1, -1)
                hazards = self._model(expanded_X.reshape(batch_size * len(times), -1),
                                      expanded_times.reshape(batch_size * len(times), -1))
                hazards = hazards.view(batch_size, n_times)
                surv_probs = torch.exp(-hazards.cumsum(dim=1))

                pred_block = torch.column_stack([
                    obs_times,
                    surv_probs
                ])

                pred_chunks.append(pred_block)
                pred_serials.append(serial_numbers)

                if (real_durations != -1).any():
                    real_durations = real_durations.to(self.device)
                    y = y.to(self.device)
                    gt_block = torch.column_stack([
                        obs_times,
                        real_durations,
                        y
                    ])
                    gt_chunks.append(gt_block)

        pred_values = torch.concat(pred_chunks, dim=0).cpu().numpy()
        serial_numbers_flat = np.concatenate(pred_serials)

        df_surv = pd.DataFrame(pred_values, columns=['time'] + times.tolist())
        df_surv.insert(0, 'id', serial_numbers_flat)
        df_surv['time'] = df_surv['time'].astype('int32')
        df_surv[times] = df_surv[times].astype('float32')

        if gt_chunks:
            gt_values = torch.concat(gt_chunks, dim=0).cpu().numpy()
            df_gt = pd.DataFrame(gt_values, columns=['time', 'duration', 'failure'])
            df_gt.insert(0, 'id', serial_numbers_flat)
            df_gt = df_gt.astype({
                'id': 'string',
                'time': 'int32',
                'duration': 'int32'
            })
            df_gt['failure'] = df_gt['failure'] == 1
        else:
            df_gt = pd.DataFrame()

        if model_state:
            self._model.train()

        return df_surv, df_gt

    def get_expected_time(self, dataloader: DataLoader, times: Optional[np.ndarray] = TIMES) -> Tuple[np.ndarray, pd.DataFrame]:
        """Computes the expected time to event for observations in the dataloader 
        based on predicted survival functions.

        Args:
            dataloader (DataLoader): A DataLoader providing data for prediction.
            times (np.ndarray, optional): Array of time points used for evaluating the survival function.
                Defaults to TIMES.

        Returns:
            Tuple[np.ndarray, pd.DataFrame]:
                - A numpy array of expected times to event for each observation.
                - A DataFrame containing ground truth durations and event indicators if available,
                  otherwise an empty DataFrame.
        """
        df_survival, df_gt = self.predict(dataloader, times=times)
        return self.get_expected_time_by_predictions(df_survival, times), df_gt

    def get_expected_time_by_predictions(self, X_pred: pd.DataFrame, times: np.ndarray) -> np.ndarray:
        """Calculates expected time to event based on predicted survival functions.

        The expected time is computed as the area under the survival curve
        for each observation using trapezoidal rule.

        Args:
            X_pred (pd.DataFrame): DataFrame containing predicted survival functions.
                Columns: ['serial_number', 'time', t1, t2, ..., tN]
            times (np.ndarray): Array of time points corresponding to the survival functions.

        Returns:
            np.ndarray: A numpy array of expected times to event for each observation.
        """
        X = X_pred
        survival_vec = X.drop(['id', 'time'], axis='columns').values
        return np.trapezoid(y=survival_vec, x=times)
