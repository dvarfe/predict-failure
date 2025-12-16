import pandas as pd
import numpy as np
from typing import Optional
from torch.utils.data import DataLoader
from ..base.model_base import AbstractModel


class DummyRandModel(AbstractModel):
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def _dataloader_to_dataframe(self, dataloader) -> pd.DataFrame:
        rows = []

        for batch in dataloader:
            serials = batch[0]
            print(serials)
            rows += serials

        return rows

    def predict(self, data, times: np.ndarray = None, id_col: Optional[str] = None) -> pd.DataFrame:
        if times is None:
            times = np.arange(0, 11)

        rows = self._dataloader_to_dataframe(data)

        n_samples = len(rows)
        n_times = len(times)

        hazards = np.random.rand(n_samples, n_times)

        cums = np.cumsum(hazards, axis=1)
        surv_np = np.exp(-cums)

        cols = [f"{t}" for t in times]
        df = pd.DataFrame(data=surv_np, index=rows, columns=cols)

        target_id = id_col or 'id'
        df[target_id] = [str(x) for x in df.index]
        # Ставим колонку с id на первое место
        if df.columns[-1] == target_id:
            df.insert(0, target_id, df.pop(target_id))

        return df
