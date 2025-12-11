import pandas as pd
import numpy as np
from typing import Optional
from ..base.model_base import AbstractModel


class DummyRandModel(AbstractModel):
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def predict(self, data: pd.DataFrame, times: np.ndarray = None, id_col: Optional[str] = None) -> pd.DataFrame:
        if times is None:
            times = np.arange(0, 11)

        n_samples = len(data)
        n_times = len(times)

        hazards = np.random.rand(n_samples, n_times)

        cums = np.cumsum(hazards, axis=1)
        surv_np = np.exp(-cums)

        cols = [f"{t}" for t in times]
        df = pd.DataFrame(data=surv_np, index=range(n_samples), columns=cols)
        if id_col is not None and id_col in data.columns:
            df['id'] = data[id_col].astype(str).values
        else:
            df['id'] = data.index.astype(str).values

        return df
