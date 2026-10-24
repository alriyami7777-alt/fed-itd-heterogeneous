"""Shared helpers for the federated ITD paper (protocol v1.0 FROZEN, 13 Oct 2026).

Splits reuse the IJACSA `_split_user_day_frame` rule verbatim:
  * chronological ("temporal"): cut at the 0.7 quantile of `day` over user-day rows;
  * user-disjoint: 70% of users to training, stratified by whether a user has any positive day.
Validation (protocol Section 3) = the same rule applied again inside the training part at 0.85,
i.e. the last 15% of the training window (chronological) or 15% of training users (user-disjoint).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

USER, DAY, LABEL = "user", "day", "label"
CLIENTS = ("cert42", "cert52", "spedia", "lanl")
PROTOCOL = {"cert42": "temporal", "cert52": "temporal", "spedia": "temporal", "lanl": "user_disjoint"}
from config import DATA


def load(name: str) -> pd.DataFrame:
    d = pd.read_parquet(DATA.format(name))
    d[LABEL] = d[LABEL].astype(int)
    return d


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in (USER, DAY, LABEL)]


def split_user_day_frame(feat: pd.DataFrame, protocol: str, train_frac=0.7, seed=0):
    """Verbatim copy of IJACSA src/train/supervised_transfer.py::_split_user_day_frame."""
    if protocol == "temporal":
        f = feat.sort_values(DAY, kind="mergesort")
        cut = f[DAY].quantile(train_frac)
        train, test = f[f[DAY] <= cut], f[f[DAY] > cut]
        if len(train) == 0 or len(test) == 0:
            n = max(1, int(len(f) * train_frac))
            train, test = f.iloc[:n], f.iloc[n:]
        return train, test
    if protocol == "user_disjoint":
        rng = np.random.default_rng(seed)
        user_pos = feat.groupby(USER)[LABEL].max()
        pos_u = user_pos[user_pos == 1].index.to_numpy()
        neg_u = user_pos[user_pos == 0].index.to_numpy()
        rng.shuffle(pos_u)
        rng.shuffle(neg_u)

        def _take(users):
            n_tr = int(round(len(users) * train_frac))
            if len(users) >= 2:
                n_tr = min(max(n_tr, 1), len(users) - 1)
            return set(users[:n_tr])

        train_users = _take(pos_u) | _take(neg_u)
        test_users = set(user_pos.index) - train_users
        return feat[feat[USER].isin(train_users)], feat[feat[USER].isin(test_users)]
    raise ValueError(protocol)


def three_way(feat: pd.DataFrame, name: str, seed: int = 0):
    """train / val / test per the frozen protocol. Chronological splits do not depend on seed."""
    proto = PROTOCOL[name]
    s = seed if proto == "user_disjoint" else 0
    trainval, test = split_user_day_frame(feat, proto, 0.7, seed=s)
    train, val = split_user_day_frame(trainval, proto, 0.85, seed=s + 1000)
    return trainval, train, val, test


def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))
