from typing import Tuple, List, Generator
from itertools import islice, cycle

import torch
from torch.utils.data import IterableDataset, get_worker_info
import numpy as np
import pandas as pd

from SP_model import TIMES

torch.manual_seed(42)
np.random.seed(42)


class DiskDataset(IterableDataset):
    def __init__(self,
                 mode: str,
                 file_paths: List[str],
                 shuffle_files: bool = True,
                 times: np.ndarray = TIMES,
                 to_cens_time_list: List[int] = [],
                 to_term_time_list: List[int] = [],
                 cens_prob: float = -1,
                 max_buffer_size: int = 5000):
        """DiskDataset constructor.

        Args:
            mode (str): Can be train, score or infer.
            root_dir (str): Directory containing the CSV files. Defaults to PREPROCESSOR_STORAGE.
            shuffle_files (bool, optional): _description_. Defaults to True.
            to_cens_time_list (List[int]): Timeshifts to past to generate new events for.
            to_term_time_list (List[int]): Timeshifts to future to generate new events for.
            cens_prob (float): probability of censoring in data. Defaults to -1, which means no over/downsampling
        """
        self._mode = mode
        self._shuffle_files = shuffle_files
        self._file_paths = file_paths
        self.times = times
        self.to_cens_time_list = to_cens_time_list
        self.to_term_time_list = to_term_time_list

        # Buffers for censored and terminal observations
        self.cens_prob = cens_prob
        self.cens_buf = []
        self.term_buf = []

        self._len = 0
        for file_path in self._file_paths:
            df = pd.read_csv(file_path)
            term = df['failure'].sum()
            self._len += df.shape[0] + term * (len(self.to_cens_time_list) +
                                               len(self.to_term_time_list)) - len(df[df['time'] == df['max_lifetime']])

    def __len__(self):
        """Returns the total number of observations in the dataset."""
        return self._len

    def __iter__(self) -> Generator[Tuple[str, int, torch.Tensor, bool, int], None, None]:
        '''
        Should always return time as the last column in data
        '''
        # Shuffle files at the start of each epoch
        worker_info = get_worker_info()
        file_paths = self._split_files_for_workers(worker_info)

        if self._shuffle_files:
            np.random.shuffle(file_paths)

        # Get data from files
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                header = lines[0].strip().split(',')
                lines = lines[1:]
                np.random.shuffle(lines)

                id_idx = header.index('serial_number')
                time_idx = header.index('time')
                if self._mode != 'infer':
                    label_idx = header.index('failure')
                    event_time_idx = header.index('max_lifetime')
                for line in lines:
                    data_line = line.strip().split(',')
                    if self._mode == 'train':
                        if data_line[event_time_idx] == data_line[time_idx]:
                            continue
                        observs = self._parse_train_line(data_line, label_idx, id_idx, time_idx, event_time_idx)
                        if self.cens_prob >= 0:
                            for observ in observs:
                                _, _, _, y, _ = observ

                                if y:
                                    if len(self.term_buf) >= self.max_buffer_size:
                                        self.term_buf.pop(0)  # удаляем самый старый
                                    self.term_buf.append(observ)

                                else:
                                    if len(self.cens_buf) >= self.max_buffer_size:
                                        self.cens_buf.pop(0)
                                    self.cens_buf.append(observ)

                            while self.term_buf and self.cens_buf:
                                if np.random.random() < self.cens_prob:
                                    yield self.cens_buf.pop(0)
                                else:
                                    yield self.term_buf.pop(0)
                        else:
                            for observ in observs:
                                yield observ
                    elif self._mode == 'score':
                        if data_line[event_time_idx] == data_line[time_idx]:
                            continue
                        yield self._parse_score_line(data_line, label_idx, id_idx, time_idx, event_time_idx)
                    elif self._mode == 'infer':
                        yield self._parse_infer_line(data_line, id_idx, time_idx)

    def _parse_train_line(self, data_line: List[str], label_idx: int, id_idx: int, time_idx: int, event_time_idx: int) -> List[Tuple[str, int, torch.Tensor, bool, int]]:
        """Parse a line of training data.

        Args:
            data_line (List[str]): A list of strings representing a line of data.
            label_idx (int): Index of the label column.
            id_idx (int): Index of the ID column.
            time_idx (int): Index of the time column.
            event_time_idx (int): Index of the event time column.

        Returns:
            Tuple[str, int, torch.Tensor, bool, int]: Parsed data including ID, time, features, label, and time to event.
        """
        # Parse the line and convert it to a tensor

        data_vec = [float(data_line[i]) for i in range(len(data_line)) if i not in [
            id_idx, time_idx, event_time_idx, label_idx]]
        cur_time = int(data_line[time_idx])
        event_time = int(data_line[event_time_idx])
        time_to_event = event_time - cur_time
        # data_vec += [time_to_event]
        y = data_line[label_idx] == '1'
        if y:
            extended_list = [[data_line[id_idx], int(data_line[time_idx]), torch.tensor(data_vec), y, time_to_event]]
            for time in self.to_cens_time_list:
                if time_to_event - time <= 0:
                    break
                extended_list.append([data_line[id_idx], int(data_line[time_idx]),
                                     torch.tensor(data_vec), 0, time_to_event - time])
            for time in self.to_term_time_list:
                extended_list.append([data_line[id_idx], int(data_line[time_idx]),
                                     torch.tensor(data_vec), 1, time_to_event + time])
            return extended_list
        else:
            return [[data_line[id_idx], int(data_line[time_idx]), torch.tensor(data_vec), y, time_to_event]]

    def _parse_score_line(self, data_line: List[str], label_idx: int, id_idx: int, time_idx: int, event_time_idx: int) -> Tuple[str, int, torch.Tensor, bool, int]:
        """Parse a line of scoring data.

        Args:
            data_line (List[str]): A list of strings representing a line of data.
            label_idx (int): Index of the label column.
            id_idx (int): Index of the ID column.
            time_idx (int): Index of the time column.
            event_time_idx (int): Index of the event time column.

        Returns:
            Tuple[str, int, torch.Tensor, bool, int]: Parsed data including ID, time, features, label, and time to event      .
        """
        data_vec = [float(data_line[i]) for i in range(len(data_line)) if i not in [
            id_idx, time_idx, event_time_idx, label_idx]]
        y = data_line[label_idx] == '1'
        cur_time = int(data_line[time_idx])
        event_time = int(data_line[event_time_idx])
        time_to_event = event_time - cur_time

        return data_line[id_idx], cur_time, torch.Tensor(data_vec), y, time_to_event

    def _parse_infer_line(self, data_line: List[str], id_idx: int, time_idx: int) -> Tuple[str, int, torch.Tensor, bool, int]:
        """Parse a line of inference data.

        Args:
            data_line (List[str]): A list of strings representing a line of data.
            id_idx (int): Index of the ID column.
            time_idx (int): Index of the time column.

        Returns:
            Tuple[str, int, torch.Tensor, bool, int]: Parsed data including ID, time, features, and placeholders for label and time to event.
        """
        data_vec = [float(data_line[i]) for i in range(len(data_line)) if i not in [id_idx, time_idx]]
        cur_time = int(data_line[time_idx])
        time_to_event = -1
        return data_line[id_idx], cur_time, torch.tensor(data_vec), 0, time_to_event

    def _split_files_for_workers(self, worker_info):
        """Split files across workers to avoid duplicates.

        Args:
            worker_info: Information about the current worker process.

        Returns:
            List[str]: A list of file paths assigned to the current worker.
        """
        # Split files across workers to avoid duplicates

        if worker_info is None:
            # Single-process mode
            return self._file_paths
        else:
            # Split files across workers
            return list(islice(
                cycle(self._file_paths),          # Create infinite cycle through files
                worker_info.id,                  # Unique index for each worker
                len(self._file_paths),            # Stop after all files are assigned
                worker_info.num_workers          # Step by total workers
            ))
