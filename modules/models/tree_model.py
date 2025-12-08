from ..base.model_base import AbstractModel
import pandas as pd
import numpy as np

class DummyTreeModel(AbstractModel):
    def fit(self, data: pd.DataFrame):
        pass

    def predict(self, data: pd.DataFrame):
        return pd.Series(np.ones(data.shape[0]))