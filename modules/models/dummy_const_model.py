import pandas as pd
import numpy as np
from ..base.model_base import AbstractModel


class DummyConstModel(AbstractModel):
    def __init__(self):
        self.hazard_ratio = 0.0
        self.surv_const = 1.0
        self.is_fitted = False

    def predict(self, data: pd.DataFrame, times: np.ndarray = None, id_col: str | None = None) -> pd.DataFrame:
        if times is None:
            times = np.arange(0, 11)

        n_samples = len(data)

        survival_functions = pd.DataFrame(
            data=self.surv_const,
            index=range(n_samples),
            columns=[f"{t}" for t in times]
        )
        if id_col is not None and id_col in data.columns:
            survival_functions['id'] = data[id_col].astype(str).values
        else:
            survival_functions['id'] = data.index.astype(str).values

        return survival_functions
