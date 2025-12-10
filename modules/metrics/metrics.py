from typing import Set

import pandas as pd
import numpy as np
from lifelines.utils import concordance_index  # type: ignore
from survivors.metrics import ibs_remain, iauc_WW_TI


def get_metrics(self, model, df_pred: pd.DataFrame, df_gt: pd.DataFrame, times: np.ndarray, metrics: Set[str], axis=-1, df_train=None):

    metrics_dict = {}

    if df_train is not None:
        iauc_train = pd.DataFrame()
        iauc_train['cens'] = df_train['failure'].astype(bool)
        iauc_train['duration'] = df_train['duration']

    survival_test = pd.DataFrame()
    survival_test['event'] = df_gt['failure'].astype(bool)
    survival_test['duration'] = df_gt['duration']

    iauc_test = pd.DataFrame()
    iauc_test['cens'] = df_gt['failure'].astype(bool)
    iauc_test['time'] = df_gt['duration']

    lifetime_pred = model.get_expected_time_by_predictions(df_pred, times)

    if 'ci' in metrics:
        ci = concordance_index(df_gt['duration'], lifetime_pred, df_gt['failure'])
        metrics_dict['ci'] = ci

    survival_estim = df_pred.drop(['id', 'time'], axis='columns')

    if 'ibs' in metrics:
        ibs = ibs_remain(
            None,
            survival_test.to_records(index=False),
            survival_estim,
            times,
            axis=axis
        )
        metrics_dict['ibs'] = ibs
    if 'ibs_bal' in metrics:

        ibs_bal = self.bal_ibs_remain(
            None,
            survival_test.to_records(index=False),
            survival_estim,
            times,
            axis=axis
        )
        metrics_dict['ibs_bal'] = ibs_bal

    if 'iauc' in metrics:
        if df_train is None:
            raise ValueError("df_train must be provided to compute iauc")
        hazard_estim = -np.log(survival_estim)
        iauc_score = iauc_WW_TI(
            iauc_train.to_records(index=False),
            iauc_test.to_records(index=False),
            hazard_estim,
            times,
        )
        metrics_dict['iauc'] = iauc_score

    return metrics_dict
