from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from torchtyping import TensorType
from sksurv.metrics import check_y_survival

MAX_CLIP = 1


class FourierTimeEncoding(nn.Module):
    def __init__(self, dim: int, log_max_time: float = 4.0):  # 10^4 ≈ 10 000
        super().__init__()
        self.register_buffer(
            "freqs", 10 ** torch.linspace(0, log_max_time, dim // 2)
        )

    def forward(self, t: TensorType[("batch", 1)]):
        ang = t * self.freqs
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class LearnableSoftplus(nn.Module):
    def __init__(self, beta_init=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta_init))

    def forward(self, x):
        return F.softplus(x, beta=self.beta.item())


class LearnableScaleParam(nn.Module):
    def __init__(self, alpha_init=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, x):
        return torch.clip(x, 0, MAX_CLIP) ** self.alpha


def ibs_like_loss(surv_pred, y_true):
    return ((surv_pred - y_true) ** 2).mean(dim=-1).mean(dim=-1).mean(dim=-1)


def masked_loss(base_criterion, surv_pred, y_true_surv, events=None, durations=None, times=None):
    if (events is None) and (durations is None):
        mse = nn.MSELoss(reduction='mean')
        return mse(surv_pred, y_true_surv)

    device = surv_pred.device

    if events is not None:
        if events.dim() > 1 and events.size(1) == 1:
            events = events.squeeze(1)
        events = events.to(device)
    if durations is not None:
        if durations.dim() > 1 and durations.size(1) == 1:
            durations = durations.squeeze(1)
        durations = durations.to(device)

    if times is None:
        raise ValueError("times must be provided when using events/durations masking")

    times = times.to(device)
    # times shape: (T,) -> make (1, T) for broadcasting
    time_points = times.view(1, -1)

    # durations -> (B,) -> (B,1) for broadcasting
    durations_f = durations.float().view(-1, 1)

    event_observed = events.to(dtype=torch.bool).view(-1)

    # valid_mask shape: (B, T)
    valid_mask = (event_observed.view(-1, 1) | (time_points <= durations_f)).to(dtype=torch.bool)

    se = (surv_pred - y_true_surv) ** 2  # shape (B, T)

    # Zero out masked (invalid) positions
    se_masked = se * valid_mask.to(dtype=se.dtype)

    valid_counts = valid_mask.sum(dim=1).to(dtype=se.dtype)  # shape (B,)

    valid_counts_safe = torch.where(valid_counts == 0, torch.ones_like(valid_counts), valid_counts)

    per_sample_mse = se_masked.sum(dim=1) / valid_counts_safe  # shape (B,)

    loss = per_sample_mse.mean()

    return loss


def ibs_remain_torch(survival_train,
                     survival_test,
                     estimate,
                     times: torch.Tensor,
                     axis: int = -1,
                     device: Optional[torch.device] = None):

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_event_np, test_time_np = check_y_survival(survival_test, allow_all_censored=True)

    if estimate.ndim == 1 and np.asarray(times).shape[0] == 1:
        estimate = estimate.reshape(-1, 1)
    est = estimate

    if torch.isinf(est).any():
        est = torch.where(torch.isinf(est), torch.zeros_like(est), est)

    test_event = torch.as_tensor(test_event_np.copy().astype(np.int64), dtype=torch.float64, device=device)
    test_time = torch.as_tensor(test_time_np.copy(), dtype=torch.float64, device=device)

    n_samples, n_times = est.shape
    if times.numel() != n_times:
        raise ValueError("length of times must match estimate.shape[1]")

    estim_before = est.pow(2) * test_event.view(-1, 1)  # (n_samples, n_times)
    estim_after = (1.0 - est).pow(2)                     # (n_samples, n_times)

    times_row = times.view(1, -1)           # (1, n_times)
    test_time_col = test_time.view(-1, 1)   # (n_samples, 1)
    mask_before = test_time_col < times_row

    brier_matrix = torch.where(mask_before, estim_before, estim_after)  # (n_samples, n_times)

    ones_col = torch.ones_like(test_event.view(-1, 1), dtype=torch.float64)
    choose_counts = torch.where(mask_before, test_event.view(-1, 1), ones_col)  # (n_samples, n_times)
    N = choose_counts.sum(dim=0)  # (n_times,)

    time_diff = (times[-1] - times[0]).item() if times[-1] > times[0] else 1.0

    if axis == -1:
        per_time_sum = brier_matrix.sum(dim=0)  # (n_times,)
        bs_per_time = torch.where(N > 0, per_time_sum / N, torch.zeros_like(per_time_sum))
        ibs = torch.trapz(bs_per_time, times) / time_diff
        return ibs

    elif axis == 0:
        ibs_per_obs = torch.trapz(brier_matrix, times, dim=1) / time_diff
        return ibs_per_obs

    elif axis == 1:
        per_time_sum = brier_matrix.sum(dim=0)  # (n_times,)
        bs_per_time = torch.where(N > 0, per_time_sum / N, torch.zeros_like(per_time_sum))
        return bs_per_time

    return None


def masked_loss(base_criterion, surv_pred, y_true_surv, events=None, durations=None, times=None):
    if (events is None) and (durations is None):
        mse = nn.MSELoss(reduction='mean')
        return mse(surv_pred, y_true_surv)

    device = surv_pred.device

    if events is not None:
        if events.dim() > 1 and events.size(1) == 1:
            events = events.squeeze(1)
        events = events.to(device)
    if durations is not None:
        if durations.dim() > 1 and durations.size(1) == 1:
            durations = durations.squeeze(1)
        durations = durations.to(device)

    if times is None:
        raise ValueError("times must be provided when using events/durations masking")

    times = times.to(device)
    # times shape: (T,) -> make (1, T) for broadcasting
    time_points = times.view(1, -1)

    # durations -> (B,) -> (B,1) for broadcasting
    durations_f = durations.float().view(-1, 1)

    event_observed = events.to(dtype=torch.bool).view(-1)

    # valid_mask shape: (B, T)
    valid_mask = (event_observed.view(-1, 1) | (time_points <= durations_f)).to(dtype=torch.bool)

    se = (surv_pred - y_true_surv) ** 2  # shape (B, T)

    # Zero out masked (invalid) positions
    se_masked = se * valid_mask.to(dtype=se.dtype)

    valid_counts = valid_mask.sum(dim=1).to(dtype=se.dtype)  # shape (B,)

    valid_counts_safe = torch.where(valid_counts == 0, torch.ones_like(valid_counts), valid_counts)

    per_sample_mse = se_masked.sum(dim=1) / valid_counts_safe  # shape (B,)

    loss = per_sample_mse.mean()

    return loss
