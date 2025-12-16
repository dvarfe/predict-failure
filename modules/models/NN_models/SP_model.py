from typing import Optional, Tuple, List
import time
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm

from .ClassifierArchitecture import ClassifierArchitecture
from .utils import ibs_remain_torch, ibs_like_loss, masked_loss
from ...base.model_base import AbstractModel

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
np.random.seed(42)

# Используем TIMES из провайдера данных или определяем по умолчанию
TIMES = np.arange(1, 10)
DEFAULT_HIDDEN_DIM = 64
DEFAULT_LR = 1e-3
DEFAULT_EPOCHS = 100
DEFAULT_DEVICE = None


class SurvPredictor(AbstractModel):
    """Survival predictor.

    Attributes:
        device (str): Device where the model and tensors are allocated ('cuda' or 'cpu').
        _model (nn.Module): Neural network model instance.
        criterion (nn.Module): Loss function used for training.
        optimizer (torch.optim.Optimizer): Optimizer for training the model.
        (TensorBoard writer removed) Optional logging not supported.
    """

    def __init__(self,
                 input_dim: Optional[int] = None,
                 hidden_dim: int = DEFAULT_HIDDEN_DIM,
                 lr: float = DEFAULT_LR,
                 epochs: int = DEFAULT_EPOCHS,
                 device: Optional[str] = DEFAULT_DEVICE):
        """Initialize survival predictor.

        Args:
            input_dim: Number of input features (will be auto-detected if None)
            hidden_dim: Number of hidden units (default: 64)
            lr: Learning rate (default: 1e-3)
            epochs: Number of training epochs (default: 100)
            device: Computation device ('cuda' or 'cpu')
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = None
        self.criterion = ibs_like_loss
        self.optimizer = None
        self.is_fitted = False

        # Training metrics
        self.loss: List[float] = []
        self.fit_times: List[float] = []
        self.best_loss = float('inf')

    def get_survival_function(self, times, event_times, status):
        surv = torch.ones((len(event_times), len(times)))
        for i, (t, s) in enumerate(zip(event_times, status)):
            surv[i, times >= t] = 0 if s else 1
        return surv

    def _detect_input_dim(self, dataloader: DataLoader):
        """Определить количество признаков из данных"""
        for batch in dataloader:
            serial_number, time, features_tensor, failure, time_to_event = batch

            self.input_dim = features_tensor.size(1)

            print(f"Определено количество признаков для SP: {self.input_dim}")
            return

        raise ValueError("Не удалось определить количество признаков из данных для SP модели")

    def fit(self,
            train_dataloader: DataLoader,
            times: Optional[np.ndarray] = None,
            id_col: Optional[str] = None) -> Optional['SurvPredictor']:
        try:
            # Используем времена по умолчанию, если не переданы
            if times is None:
                times = TIMES

            # Определяем input_dim из данных, если он не задан
            if self.input_dim is None:
                self._detect_input_dim(train_dataloader)

            # Инициализируем модель, если она еще не создана
            if self._model is None:
                self._model = nn.DataParallel(ClassifierArchitecture(self.input_dim, self.hidden_dim).to(self.device))
                self.optimizer = torch.optim.AdamW(self._model.parameters(), lr=self.lr)

            self._model.train()
            times_tensor = torch.as_tensor(times, device=self.device, dtype=torch.float32)
            global_step = 0

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

                        tepoch.set_postfix(loss=loss_value)

                # Epoch statistics
                avg_epoch_loss = epoch_loss / epoch_steps
                epoch_time = time.time() - start_time
                self.loss.append(avg_epoch_loss)
                self.fit_times.append(epoch_time)

                # logging removed

            self.is_fitted = True
            return self
        except Exception as e:
            print(f"Ошибка во время обучения модели: {e}")
            return None

    def predict(self, dataloader: DataLoader,
                times: Optional[np.ndarray] = TIMES,
                id_col: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Predicts survival functions for observations from the dataloader.

         For each observation, this method predicts the survival probability
         over a predefined set of time points using the trained model.

         Args:
             dataloader (DataLoader): A DataLoader providing batches of data in the form:
                 (serial_numbers, obs_times, X, y, real_durations)
             times (np.ndarray, optional): Array of time points at which the survival function is evaluated.
                 Defaults to TIMES.
             id_col (str, optional): Name of the ID column for output DataFrame.

         Returns:
             Tuple[pd.DataFrame, pd.DataFrame]:
                 - A DataFrame containing predicted survival functions for each observation.
                   Columns: [id_col, 'time', t1, t2, ..., tN]
                 - A DataFrame containing ground truth durations and event indicators if available,
                   otherwise an empty DataFrame.
         """
        try:
            model_state = self._model.training
            self._model.eval()
            pred_chunks = []
            pred_serials = []
            gt_chunks = []

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
            ids = serial_numbers_flat.astype(str)
            target_id = id_col or 'id'
            df_surv.insert(0, target_id, ids)
            df_surv.index = ids
            df_surv['time'] = df_surv['time'].astype('int32')
            df_surv[times] = df_surv[times].astype('float32')

            if gt_chunks:
                gt_values = torch.concat(gt_chunks, dim=0).cpu().numpy()
                df_gt = pd.DataFrame(gt_values, columns=['time', 'duration', 'failure'])
                df_gt.insert(0, target_id, serial_numbers_flat)
                df_gt = df_gt.astype({
                    target_id: 'string',
                    'time': 'int32',
                    'duration': 'int32'
                })
                df_gt['failure'] = df_gt['failure'] == 1
            else:
                df_gt = pd.DataFrame()

            if model_state:
                self._model.train()

            return df_surv

        except Exception as e:
            print(f"Ошибка во время предсказания: {e}")
            return None

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
                  otherwise an empty DataFrame. Returns (None, None) if prediction fails.
        """
        try:
            df_survival, df_gt = self.predict(dataloader, times=times)
            if df_survival is None:
                return None
            return self.get_expected_time_by_predictions(df_survival, times), df_gt
        except Exception as e:
            print(f"Ошибка во время вычисления ожидаемого времени: {e}")
            return None

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
        # Detect id column: first non-numeric column (excluding 'time')
        id_col_local = None
        for c in X.columns:
            if c == 'time':
                continue
            try:
                float(c)
                # numeric column (time)
                continue
            except Exception:
                id_col_local = c
                break

        if id_col_local is None:
            # fallback: if there's a column literally named 'id', use it
            id_col_local = 'id' if 'id' in X.columns else None

        drop_cols = ['time'] + ([id_col_local] if id_col_local is not None else [])
        survival_vec = X.drop(drop_cols, axis='columns').values
        return np.trapezoid(y=survival_vec, x=times)
