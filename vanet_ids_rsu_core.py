from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    _LIGHTGBM_IMPORT_ERROR: Optional[Exception] = None
except ModuleNotFoundError as exc:
    lgb = None
    _LIGHTGBM_IMPORT_ERROR = exc

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Bidirectional, Dense, Dropout, LSTM, LayerNormalization
    from tensorflow.keras.models import Sequential, load_model
    _TENSORFLOW_IMPORT_ERROR: Optional[Exception] = None
except ModuleNotFoundError as exc:
    tf = None
    EarlyStopping = None
    Bidirectional = None
    Dense = None
    Dropout = None
    LSTM = None
    LayerNormalization = None
    Sequential = None
    load_model = None
    _TENSORFLOW_IMPORT_ERROR = exc


ARTIFACT_FAMILY = "rsu_multi_head_v3"
ARTIFACT_VERSION = "3.0.0"
LEGACY_FAMILY_ARTIFACT = "legacy_family_runtime"
ARCHIVED_FAMILY_ENSEMBLE_ARTIFACT = "archived_family_ensemble_v1"
SCENARIO_LGBM_ARTIFACT = "scenario_lgbm_v1"
DEFAULT_WINDOW_SIZE = 15
DEFAULT_SEQ_LEN = 20
HEAD_VECTOR_ORDER = [
    "general",
    "pos_speed",
    "replay_stale",
    "dos",
    "dos_iforest",
    "sybil",
    "integrity",
]
OBU_FLAG_COLUMNS = [
    "flag_speed_phys",
    "flag_acc_phys",
    "flag_hr_phys",
    "flag_consistency",
    "flag_proto_nan",
    "flag_dt_nonpos",
]
STRICT_REQUIRED_HEADS = ["general", "pos_speed", "replay_stale", "dos", "dos_iforest", "sybil", "integrity"]
STRICT_REQUIRED_MODEL_HEADS = ["general", "pos_speed", "dos", "sybil", "integrity"]
DEFAULT_OBU_THRESHOLDS = {
    "speed_abs_max": 80.0,
    "acc_abs_max": 12.0,
    "heading_rate_abs_max": 2.0,
    "consistency_err_max": 5.0,
    "dt_min": 0.0,
    "dt_max": 2.0,
}
NON_FEATURE_COLUMNS = {
    "row_id",
    "row_key",
    "file_path",
    "receiver_pseudo",
    "sender_pseudo",
    "t_prev",
    "t_curr",
    "creation_time",
    "attack_type",
    "meta_attack_type",
    "label",
    "attack_id",
}
ATTACK_TYPES = {
    0: "Genuine",
    1: "ConstPos",
    2: "ConstPosOffset",
    3: "RandomPos",
    4: "RandomPosOffset",
    5: "ConstSpeed",
    6: "ConstSpeedOffset",
    7: "RandomSpeed",
    8: "RandomSpeedOffset",
    9: "EventualStop",
    10: "Disruptive",
    11: "DataReplay",
    12: "StaleMessages",
    13: "DoS",
    14: "DoSRandom",
    15: "DoSDisruptive",
    16: "GridSybil",
    17: "DataReplaySybil",
    18: "DoSRandomSybil",
    19: "DoSDisruptiveSybil",
}
ATTACK_NAME_TO_ID = {
    name.lower(): attack_id for attack_id, name in ATTACK_TYPES.items()
}
ATTACK_FAMILIES = {
    "pos_speed": {1, 2, 3, 4, 5, 6, 7, 8, 9},
    "replay_stale": {11, 12, 17},
    "dos": {13, 14, 15, 18, 19},
    "sybil": {16, 17, 18, 19},
}
TRAIN_FAMILY_CHOICES = {"binary", "all", "pos_speed", "replay_stale", "dos", "sybil"}
ARCHIVED_FAMILY_ORDER = ("pos_speed", "replay_stale", "dos", "sybil")
PREFERRED_ARCHIVED_FAMILY_REVISIONS = {
    "pos_speed": "20250922-141757",
}


def log(message: str) -> None:
    print(message, flush=True)


def ensure_dir(path: str | os.PathLike[str]) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def _missing_dependency_message(package: str, context: str) -> str:
    return (
        f"{package} is required for {context}.\n"
        f"Install it in this Python environment with:\n"
        f"  {sys.executable} -m pip install {package}"
    )


def require_lightgbm(context: str) -> None:
    if lgb is None:
        raise ModuleNotFoundError(_missing_dependency_message("lightgbm", context)) from _LIGHTGBM_IMPORT_ERROR


def require_tensorflow(context: str) -> None:
    if tf is None:
        raise ModuleNotFoundError(_missing_dependency_message("tensorflow", context)) from _TENSORFLOW_IMPORT_ERROR


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cleaned = []
    for col in df.columns:
        value = str(col).replace("\ufeff", "").strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1].strip()
        cleaned.append(value)
    df.columns = cleaned
    return df


def sanitize_entity_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in (
        "sender_pseudo",
        "receiver_pseudo",
        "mb_version",
        "attack_type",
        "meta_attack_type",
    ):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.strip("'\"")
    return df


def minmax01(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=float)
    lo = float(finite.min())
    hi = float(finite.max())
    if abs(hi - lo) <= 1e-12:
        return np.zeros_like(arr, dtype=float)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def robust_abs_percentile(series: pd.Series, percentile: float, default: float) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().abs()
    if values.empty:
        return float(default)
    return float(np.percentile(values.to_numpy(dtype=float), percentile))


def thesis_obu_thresholds() -> Dict[str, float]:
    return dict(DEFAULT_OBU_THRESHOLDS)


def is_binary_feature_series(series: pd.Series) -> bool:
    values = (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )
    if values.size == 0:
        return False
    unique = np.unique(values)
    return bool(unique.size <= 2 and set(unique.tolist()).issubset({0.0, 1.0}))


def fit_feature_scaler(aligned_df: pd.DataFrame, feature_config: Dict[str, Any]) -> Optional[StandardScaler]:
    continuous_columns = list(feature_config.get("continuous_feature_columns", []))
    if not continuous_columns:
        return None
    scaler = StandardScaler()
    scaler.fit(aligned_df[continuous_columns].to_numpy(dtype=float))
    return scaler


def transform_feature_matrix(
    aligned_df: pd.DataFrame,
    feature_config: Dict[str, Any],
    scaler: Optional[StandardScaler],
) -> pd.DataFrame:
    transformed = aligned_df.copy()
    continuous_columns = list(feature_config.get("continuous_feature_columns", []))
    if scaler is not None and continuous_columns:
        transformed.loc[:, continuous_columns] = scaler.transform(
            transformed[continuous_columns].to_numpy(dtype=float)
        )
    return transformed


def best_f1_threshold(y_true: Sequence[int], scores: Sequence[float], default: float = 0.5) -> float:
    y_arr = np.asarray(y_true, dtype=int)
    s_arr = np.asarray(scores, dtype=float)
    if len(np.unique(y_arr)) < 2:
        return float(default)
    prec, rec, thr = precision_recall_curve(y_arr, s_arr)
    if len(thr) == 0:
        return float(default)
    f1 = (2.0 * prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-9)
    return float(thr[int(np.nanargmax(f1))])


def angle_normalize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    values = (values + np.pi) % (2.0 * np.pi) - np.pi
    return pd.Series(values, index=series.index)


def build_lstm(input_shape: Tuple[int, int]) -> tf.keras.Model:
    require_tensorflow("the replay_stale LSTM head")
    model = Sequential(
        [
            Bidirectional(LSTM(64, return_sequences=True, activation="tanh"), input_shape=input_shape),
            LayerNormalization(),
            Dropout(0.2),
            Bidirectional(LSTM(64, activation="tanh")),
            LayerNormalization(),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
    return model


def attack_name_to_id(value: Any) -> int:
    if value is None:
        return 0
    text = str(value).strip().strip("'\"").lower()
    if text.isdigit():
        return int(text)
    return ATTACK_NAME_TO_ID.get(text, 0)


def parse_f2md_bsm_file(path: str | os.PathLike[str]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        obj = json.load(handle)

    bp = obj.get("BsmPrint", {})
    meta = bp.get("Metadata", {})
    bsms = bp.get("BSMs", []) or []
    if not bsms:
        raise ValueError(f"No BSM payload in {path}")
    bsm = bsms[0]

    sender = bsm.get("Pseudonym") or bsm.get("RealId") or "unknown_sender"
    receiver = meta.get("receiverPseudo") or meta.get("receiverId") or "unknown_receiver"
    attack_type = bsm.get("AttackType") or meta.get("attackType") or "Genuine"
    meta_attack_type = meta.get("attackType") or attack_type
    attack_id = attack_name_to_id(attack_type) or attack_name_to_id(meta_attack_type)
    label = int(attack_id != 0)

    pos = bsm.get("Pos", [0, 0, 0])
    spd = bsm.get("Speed", [0, 0, 0])
    acc = bsm.get("Accel", [0, 0, 0])
    heading = bsm.get("Heading", [1, 0, 0])
    pos_conf = bsm.get("PosConfidence", [0, 0, 0])
    spd_conf = bsm.get("SpeedConfidence", [0, 0, 0])
    acc_conf = bsm.get("AccelConfidence", [0, 0, 0])
    head_conf = bsm.get("HeadingConfidence", [0, 0, 0])
    vx = float(spd[0]) if len(spd) > 0 else 0.0
    vy = float(spd[1]) if len(spd) > 1 else 0.0
    ax = float(acc[0]) if len(acc) > 0 else 0.0
    ay = float(acc[1]) if len(acc) > 1 else 0.0
    heading_angle = math.atan2(float(heading[1]) if len(heading) > 1 else 0.0, float(heading[0]) if len(heading) > 0 else 1.0)

    return {
        "file_path": str(path),
        "receiver_pseudo": str(receiver),
        "sender_pseudo": str(sender),
        "t_curr": float(bsm.get("CreationTime", meta.get("generationTime", 0.0)) or 0.0),
        "creation_time": float(bsm.get("CreationTime", meta.get("generationTime", 0.0)) or 0.0),
        "x_curr": float(pos[0]) if len(pos) > 0 else 0.0,
        "y_curr": float(pos[1]) if len(pos) > 1 else 0.0,
        "vx": vx,
        "vy": vy,
        "ax": ax,
        "ay": ay,
        "speed_curr": float(math.hypot(vx, vy)),
        "acc_curr": float(math.hypot(ax, ay)),
        "heading_curr": heading_angle,
        "pos_conf_x_curr": float(pos_conf[0]) if len(pos_conf) > 0 else 0.0,
        "pos_conf_y_curr": float(pos_conf[1]) if len(pos_conf) > 1 else 0.0,
        "spd_conf_x_curr": float(spd_conf[0]) if len(spd_conf) > 0 else 0.0,
        "spd_conf_y_curr": float(spd_conf[1]) if len(spd_conf) > 1 else 0.0,
        "acc_conf_x_curr": float(acc_conf[0]) if len(acc_conf) > 0 else 0.0,
        "acc_conf_y_curr": float(acc_conf[1]) if len(acc_conf) > 1 else 0.0,
        "head_conf_x_curr": float(head_conf[0]) if len(head_conf) > 0 else 0.0,
        "head_conf_y_curr": float(head_conf[1]) if len(head_conf) > 1 else 0.0,
        "mb_version": str(bsm.get("MbType") or meta.get("mbType") or "V2"),
        "attack_type": str(attack_type),
        "meta_attack_type": str(meta_attack_type),
        "attack_id": int(attack_id),
        "label": int(label),
    }


def load_raw_bsm_directory(root: str | os.PathLike[str]) -> pd.DataFrame:
    root_path = Path(root)
    files = sorted(root_path.rglob("*.bsm"))
    records = [parse_f2md_bsm_file(path) for path in files]
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def _stable_row_key(df: pd.DataFrame) -> pd.Series:
    if "row_key" in df.columns:
        return df["row_key"].astype(str)
    if "file_path" in df.columns:
        return df["file_path"].astype(str)

    parts = []
    for col in ("receiver_pseudo", "sender_pseudo", "t_curr", "x_curr", "y_curr", "speed_curr"):
        if col in df.columns:
            parts.append(df[col].astype(str))
    if not parts:
        return pd.Series(np.arange(len(df)).astype(str), index=df.index)
    return parts[0].str.cat(parts[1:], sep="|")


def normalize_bsm_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = sanitize_entity_values(sanitize_columns(raw_df))
    df = df.copy()

    if "row_id" not in df.columns:
        df["row_id"] = np.arange(len(df), dtype=int)
    df["row_key"] = _stable_row_key(df)

    if "t_curr" not in df.columns:
        if "creation_time" in df.columns:
            df["t_curr"] = pd.to_numeric(df["creation_time"], errors="coerce").fillna(0.0)
        elif "time" in df.columns:
            df["t_curr"] = pd.to_numeric(df["time"], errors="coerce").fillna(0.0)
        else:
            df["t_curr"] = np.arange(len(df), dtype=float)
    df["t_curr"] = pd.to_numeric(df["t_curr"], errors="coerce").fillna(0.0)

    if "x_curr" not in df.columns and "x" in df.columns:
        df["x_curr"] = pd.to_numeric(df["x"], errors="coerce").fillna(0.0)
    if "y_curr" not in df.columns and "y" in df.columns:
        df["y_curr"] = pd.to_numeric(df["y"], errors="coerce").fillna(0.0)

    if "speed_curr" not in df.columns:
        if {"vx", "vy"}.issubset(df.columns):
            vx = pd.to_numeric(df["vx"], errors="coerce").fillna(0.0)
            vy = pd.to_numeric(df["vy"], errors="coerce").fillna(0.0)
            df["speed_curr"] = np.hypot(vx, vy)
        else:
            df["speed_curr"] = 0.0

    if "acc_curr" not in df.columns:
        if {"ax", "ay"}.issubset(df.columns):
            ax = pd.to_numeric(df["ax"], errors="coerce").fillna(0.0)
            ay = pd.to_numeric(df["ay"], errors="coerce").fillna(0.0)
            df["acc_curr"] = np.hypot(ax, ay)
        else:
            df["acc_curr"] = 0.0

    if "heading_curr" not in df.columns and "heading" in df.columns:
        df["heading_curr"] = pd.to_numeric(df["heading"], errors="coerce").fillna(0.0)
    if "heading_curr" not in df.columns:
        df["heading_curr"] = 0.0

    for source, target in (
        ("pos_conf_x", "pos_conf_x_curr"),
        ("pos_conf_y", "pos_conf_y_curr"),
        ("spd_conf_x", "spd_conf_x_curr"),
        ("spd_conf_y", "spd_conf_y_curr"),
        ("acc_conf_x", "acc_conf_x_curr"),
        ("acc_conf_y", "acc_conf_y_curr"),
        ("head_conf_x", "head_conf_x_curr"),
        ("head_conf_y", "head_conf_y_curr"),
    ):
        if target not in df.columns and source in df.columns:
            df[target] = pd.to_numeric(df[source], errors="coerce").fillna(0.0)

    for col in (
        "x_curr",
        "y_curr",
        "speed_curr",
        "acc_curr",
        "heading_curr",
        "pos_conf_x_curr",
        "pos_conf_y_curr",
        "spd_conf_x_curr",
        "spd_conf_y_curr",
        "acc_conf_x_curr",
        "acc_conf_y_curr",
        "head_conf_x_curr",
        "head_conf_y_curr",
    ):
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for col in ("sender_pseudo", "receiver_pseudo"):
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = df[col].astype(str)

    if "attack_id" not in df.columns:
        if "attack_type" in df.columns:
            df["attack_id"] = df["attack_type"].map(attack_name_to_id).fillna(0).astype(int)
        else:
            df["attack_id"] = 0
    else:
        df["attack_id"] = pd.to_numeric(df["attack_id"], errors="coerce").fillna(0).astype(int)

    if "label" not in df.columns:
        if "attack_id" in df.columns:
            df["label"] = (df["attack_id"] != 0).astype(int)
        else:
            df["label"] = 0
    else:
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)

    sender_col = df["sender_pseudo"].astype(str)
    df = df.sort_values(["sender_pseudo", "t_curr", "row_id"], kind="mergesort").reset_index(drop=True)
    grp = df.groupby("sender_pseudo", sort=False)

    if "t_prev" not in df.columns:
        df["t_prev"] = grp["t_curr"].shift(1).fillna(df["t_curr"])
    if "x_prev" not in df.columns:
        df["x_prev"] = grp["x_curr"].shift(1).fillna(df["x_curr"])
    if "y_prev" not in df.columns:
        df["y_prev"] = grp["y_curr"].shift(1).fillna(df["y_curr"])
    if "speed_prev" not in df.columns:
        df["speed_prev"] = grp["speed_curr"].shift(1).fillna(df["speed_curr"])
    if "acc_prev" not in df.columns:
        df["acc_prev"] = grp["acc_curr"].shift(1).fillna(df["acc_curr"])
    if "heading_prev" not in df.columns:
        df["heading_prev"] = grp["heading_curr"].shift(1).fillna(df["heading_curr"])

    if "dt" not in df.columns:
        df["dt"] = (df["t_curr"] - df["t_prev"]).clip(lower=0.0)
    if "dx" not in df.columns:
        df["dx"] = df["x_curr"] - df["x_prev"]
    if "dy" not in df.columns:
        df["dy"] = df["y_curr"] - df["y_prev"]
    if "dist" not in df.columns:
        df["dist"] = np.hypot(df["dx"], df["dy"])
    if "dv" not in df.columns:
        df["dv"] = df["speed_curr"] - df["speed_prev"]
    if "dacc" not in df.columns:
        df["dacc"] = df["acc_curr"] - df["acc_prev"]

    df["heading_prev"] = pd.to_numeric(df["heading_prev"], errors="coerce").fillna(0.0)
    df["heading_curr"] = pd.to_numeric(df["heading_curr"], errors="coerce").fillna(0.0)
    df["dtheta"] = angle_normalize(df["heading_curr"] - df["heading_prev"])
    return df.reset_index(drop=True)


def _rolling_ratio(flags: pd.Series, groups: pd.Series, window_size: int) -> pd.Series:
    return (
        flags.astype(float)
        .groupby(groups, sort=False)
        .transform(lambda s: s.rolling(window=window_size, min_periods=2).mean())
        .fillna(0.0)
    )


def feature_engineering(df: pd.DataFrame, window_size: int = DEFAULT_WINDOW_SIZE) -> pd.DataFrame:
    out = normalize_bsm_dataframe(df)
    out = out.sort_values(["sender_pseudo", "t_curr", "row_id"], kind="mergesort").reset_index(drop=True)
    groups = out.groupby("sender_pseudo", sort=False)

    out["dt"] = pd.to_numeric(out["dt"], errors="coerce").fillna(0.0).clip(lower=0.0)
    out["speed_prev"] = pd.to_numeric(out["speed_prev"], errors="coerce").fillna(out["speed_curr"])
    out["speed_curr"] = pd.to_numeric(out["speed_curr"], errors="coerce").fillna(0.0)
    out["dv"] = out["speed_curr"] - out["speed_prev"]
    denom_dt = out["dt"].replace(0.0, np.nan)

    out["acc_curr"] = pd.to_numeric(out["acc_curr"], errors="coerce")
    out["acc_curr"] = out["acc_curr"].fillna(out["dv"] / denom_dt)
    out["acc_curr"] = out["acc_curr"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    out["acc_prev"] = pd.to_numeric(out["acc_prev"], errors="coerce")
    out["acc_prev"] = out["acc_prev"].fillna(groups["acc_curr"].shift(1))
    out["acc_prev"] = out["acc_prev"].replace([np.inf, -np.inf], 0.0).fillna(out["acc_curr"])
    out["dacc"] = out["acc_curr"] - out["acc_prev"]
    out["dacc_jerk"] = out["dacc"] / denom_dt
    out["dacc_jerk"] = out["dacc_jerk"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    out["jerk"] = out["dacc_jerk"]

    out["heading_prev"] = groups["heading_curr"].shift(1).fillna(out["heading_prev"])
    out["heading_prev"] = pd.to_numeric(out["heading_prev"], errors="coerce").fillna(out["heading_curr"])
    out["dtheta"] = angle_normalize(out["heading_curr"] - out["heading_prev"])
    out["heading_rate"] = (out["dtheta"] / denom_dt).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    out["dx"] = pd.to_numeric(out["dx"], errors="coerce").fillna(out["x_curr"] - out["x_prev"])
    out["dy"] = pd.to_numeric(out["dy"], errors="coerce").fillna(out["y_curr"] - out["y_prev"])
    out["dist"] = pd.to_numeric(out["dist"], errors="coerce").fillna(np.hypot(out["dx"], out["dy"]))

    cos_h = np.cos(out["heading_prev"])
    sin_h = np.sin(out["heading_prev"])
    x_pred = out["x_prev"] + out["speed_prev"] * cos_h * out["dt"]
    y_pred = out["y_prev"] + out["speed_prev"] * sin_h * out["dt"]
    out["dr_dx"] = out["x_curr"] - x_pred
    out["dr_dy"] = out["y_curr"] - y_pred
    out["dr_angle"] = np.arctan2(out["dr_dy"], out["dr_dx"])
    out["sin_a"] = np.sin(out["dr_angle"])
    out["cos_a"] = np.cos(out["dr_angle"])

    mean_cos = (
        out["cos_a"]
        .groupby(out["sender_pseudo"], sort=False)
        .transform(lambda s: s.rolling(window_size, min_periods=2).mean())
        .fillna(0.0)
    )
    mean_sin = (
        out["sin_a"]
        .groupby(out["sender_pseudo"], sort=False)
        .transform(lambda s: s.rolling(window_size, min_periods=2).mean())
        .fillna(0.0)
    )
    out[f"dr_angle_var_w{window_size}"] = 1.0 - np.sqrt(mean_cos**2 + mean_sin**2)

    out["neg_acc_flag"] = (out["acc_curr"] < -0.30).astype(int)
    out["low_speed_flag"] = (out["speed_curr"] < 0.50).astype(int)
    out[f"neg_acc_ratio_w{window_size}"] = _rolling_ratio(out["neg_acc_flag"], out["sender_pseudo"], window_size)
    out[f"low_speed_ratio_w{window_size}"] = _rolling_ratio(out["low_speed_flag"], out["sender_pseudo"], window_size)

    eff = max(1, window_size - 1)
    span = groups["t_curr"].transform(lambda s: s - s.shift(eff))
    out["rate_msgs_per_s"] = eff / span.replace(0.0, np.nan)
    out["rate_msgs_per_s"] = (
        out["rate_msgs_per_s"]
        .replace([np.inf, -np.inf], np.nan)
        .groupby(out["sender_pseudo"], sort=False)
        .transform(lambda s: s.bfill().ffill())
        .fillna(0.0)
    )

    for base_col in ("dv", "dacc_jerk", "heading_rate", "dist", "dt"):
        roll = out.groupby("sender_pseudo", sort=False)[base_col].rolling(window_size, min_periods=2)
        out[f"{base_col}_mean_w{window_size}"] = roll.mean().reset_index(level=0, drop=True).fillna(0.0)
        out[f"{base_col}_std_w{window_size}"] = roll.std().reset_index(level=0, drop=True).fillna(0.0)
        out[f"{base_col}_max_w{window_size}"] = roll.max().reset_index(level=0, drop=True).fillna(0.0)

    out[f"dt_jitter_w{window_size}"] = (
        out.groupby("sender_pseudo", sort=False)["dt"]
        .rolling(window_size, min_periods=2)
        .std()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )
    out[f"freeze_ratio_dv_w{window_size}"] = _rolling_ratio(out["dv"].abs() < 1e-4, out["sender_pseudo"], window_size)
    out[f"freeze_ratio_dist_w{window_size}"] = _rolling_ratio(out["dist"].abs() < 1e-3, out["sender_pseudo"], window_size)
    out[f"freeze_ratio_hr_w{window_size}"] = _rolling_ratio(out["heading_rate"].abs() < 1e-4, out["sender_pseudo"], window_size)

    out["consistency_err"] = (out["dist"] - out["speed_curr"] * out["dt"]).abs()
    out[f"consistency_err_mean_w{window_size}"] = (
        out.groupby("sender_pseudo", sort=False)["consistency_err"]
        .rolling(window_size, min_periods=2)
        .mean()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )

    state_hash = (
        out["x_curr"].round(3).astype(str)
        + "|"
        + out["y_curr"].round(3).astype(str)
        + "|"
        + out["speed_curr"].round(2).astype(str)
        + "|"
        + out["heading_curr"].round(2).astype(str)
    )
    out["state_hash"] = state_hash

    def _state_code(series: pd.Series) -> pd.Series:
        codes, _ = pd.factorize(series, sort=False)
        return pd.Series(codes.astype("int64"), index=series.index)

    out["state_code"] = out.groupby("sender_pseudo", sort=False)["state_hash"].transform(_state_code)

    def _dup_ratio(window: np.ndarray) -> float:
        if window.size == 0:
            return 0.0
        return 1.0 - (np.unique(window.astype("int64")).size / max(1, window.size))

    out["state_dup_ratio_w"] = (
        out.groupby("sender_pseudo", sort=False)["state_code"]
        .transform(lambda s: s.rolling(window_size, min_periods=2).apply(_dup_ratio, raw=True))
        .fillna(0.0)
    )

    median_dt = groups["dt"].transform(lambda s: s.rolling(window_size, min_periods=2).median()).fillna(0.0)
    mad_dt = groups["dt"].transform(
        lambda s: s.rolling(window_size, min_periods=2).apply(
            lambda w: np.median(np.abs(w - np.median(w))) if len(w) else 0.0,
            raw=True,
        )
    ).fillna(0.0) + 1e-6
    out["dt_z"] = ((out["dt"] - median_dt) / mad_dt).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    mean_dt = groups["dt"].transform(lambda s: s.rolling(window_size, min_periods=2).mean()).fillna(0.0)
    std_dt = groups["dt"].transform(lambda s: s.rolling(window_size, min_periods=2).std()).fillna(0.0)
    out["dt_cv_w"] = (std_dt / (mean_dt + 1e-6)).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    out["rate_ewma"] = (
        out.groupby("sender_pseudo", sort=False)["rate_msgs_per_s"]
        .transform(lambda s: s.ewm(span=window_size, adjust=False).mean())
        .fillna(0.0)
    )
    residual = out["rate_msgs_per_s"] - out["rate_ewma"]

    def _cusum(series: pd.Series) -> pd.Series:
        total = 0.0
        values = []
        for value in series.to_numpy(dtype=float):
            total = max(0.0, total + value)
            values.append(total)
        return pd.Series(values, index=series.index)

    out["rate_cusum_pos"] = residual.groupby(out["sender_pseudo"], sort=False).transform(_cusum).fillna(0.0)
    out["rate_cusum_neg"] = (-residual).groupby(out["sender_pseudo"], sort=False).transform(_cusum).fillna(0.0)

    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def add_sybil_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["window_id"] = np.floor(pd.to_numeric(out["t_curr"], errors="coerce").fillna(0.0) / 5.0).astype(int)
    grp = out.groupby("window_id", sort=True)

    unique_ids = grp["sender_pseudo"].nunique()
    out["sybil_unique_ids_5s"] = out["window_id"].map(unique_ids).fillna(0.0)

    def _entropy(series: pd.Series) -> float:
        probs = series.value_counts(normalize=True)
        return float(-(probs * np.log(probs + 1e-9)).sum())

    entropy = grp["sender_pseudo"].apply(_entropy)
    out["sybil_sender_entropy_5s"] = out["window_id"].map(entropy).fillna(0.0)

    window_sets = grp["sender_pseudo"].apply(lambda s: set(s.astype(str).tolist()))
    jaccard = {}
    for window_id, current in window_sets.items():
        previous = window_sets.get(window_id - 1, set())
        denom = len(current | previous)
        jaccard[window_id] = 0.0 if denom == 0 else len(current & previous) / denom
    out["sybil_jaccard_ids_5s"] = out["window_id"].map(jaccard).fillna(0.0)

    first_window = out.groupby("sender_pseudo", sort=False)["window_id"].transform("min")
    new_id_flag = (out["window_id"] == first_window).astype(int)
    new_rate = new_id_flag.groupby(out["window_id"], sort=True).transform("mean")
    out["sybil_new_ids_rate"] = new_rate.fillna(0.0)

    rate_by_window = out.groupby("window_id", sort=True)["sybil_new_ids_rate"].first().sort_index()
    ewma = rate_by_window.ewm(span=8, adjust=False).mean()
    burst = (rate_by_window - ewma).fillna(0.0)
    out["sybil_new_ids_burst"] = out["window_id"].map(burst).fillna(0.0)
    return out


def derive_obu_thresholds(df_normal: pd.DataFrame) -> Dict[str, float]:
    _ = df_normal
    return thesis_obu_thresholds()


def add_obu_evidence_flags(df: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    proto_cols = [c for c in ("dt", "speed_curr", "acc_curr", "heading_curr") if c in out.columns]
    if proto_cols:
        out["flag_proto_nan"] = (
            out[proto_cols].replace([np.inf, -np.inf], np.nan).isna().any(axis=1).astype(int)
        )
    else:
        out["flag_proto_nan"] = 0

    out["flag_dt_nonpos"] = (pd.to_numeric(out["dt"], errors="coerce").fillna(0.0) <= float(thresholds["dt_min"])).astype(int)
    out["flag_dt_large"] = (pd.to_numeric(out["dt"], errors="coerce").fillna(0.0) > float(thresholds["dt_max"])).astype(int)
    out["flag_speed_phys"] = (
        pd.to_numeric(out["speed_curr"], errors="coerce").fillna(0.0).abs() > float(thresholds["speed_abs_max"])
    ).astype(int)
    out["flag_acc_phys"] = (
        pd.to_numeric(out["acc_curr"], errors="coerce").fillna(0.0).abs() > float(thresholds["acc_abs_max"])
    ).astype(int)
    out["flag_hr_phys"] = (
        pd.to_numeric(out["heading_rate"], errors="coerce").fillna(0.0).abs() > float(thresholds["heading_rate_abs_max"])
    ).astype(int)
    out["flag_consistency"] = (
        pd.to_numeric(out["consistency_err"], errors="coerce").fillna(0.0) > float(thresholds["consistency_err_max"])
    ).astype(int)
    out["proto_anom_count"] = out[["flag_proto_nan", "flag_dt_nonpos", "flag_dt_large"]].sum(axis=1)
    out["obu_flag_count"] = out[OBU_FLAG_COLUMNS].sum(axis=1)
    out["obu_risk"] = out["obu_flag_count"] / float(len(OBU_FLAG_COLUMNS))
    out["obu_anom"] = (out["obu_flag_count"] > 0).astype(int)
    return out


def prepare_thesis_dataframe(
    raw_df: pd.DataFrame,
    *,
    window_size: int = DEFAULT_WINDOW_SIZE,
    obu_thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    prepared = feature_engineering(raw_df, window_size=window_size)
    if obu_thresholds is None:
        obu_thresholds = thesis_obu_thresholds()
    prepared = add_obu_evidence_flags(prepared, obu_thresholds)
    prepared = add_sybil_features(prepared)
    numeric_cols = prepared.select_dtypes(include=[np.number]).columns.tolist()
    prepared[numeric_cols] = prepared[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return prepared, obu_thresholds


def family_labels(df: pd.DataFrame, family: str) -> pd.Series:
    if family == "integrity":
        return df["label"].astype(int)
    ids = pd.to_numeric(df.get("attack_id", 0), errors="coerce").fillna(0).astype(int)
    return ids.isin(ATTACK_FAMILIES.get(family, set())).astype(int)


def select_general_features(df: pd.DataFrame) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if col not in NON_FEATURE_COLUMNS]


def select_pos_speed_features(feature_columns: Sequence[str]) -> List[str]:
    selected = []
    for col in feature_columns:
        if (
            col.startswith(("x_", "y_", "dr_", "heading_", "speed_", "pos_conf_", "spd_conf_"))
            or col in {"dx", "dy", "dist", "dv", "dtheta", "acc_curr", "acc_prev", "dacc", "jerk", "consistency_err"}
        ):
            selected.append(col)
    return selected or list(feature_columns)


def select_dos_features(feature_columns: Sequence[str]) -> List[str]:
    selected = []
    for col in feature_columns:
        if (
            col.startswith("rate_")
            or col.startswith("dt_")
            or col.startswith("freeze_ratio")
            or col.startswith("state_dup_ratio")
            or col in {"rate_msgs_per_s", "window_id"}
        ):
            selected.append(col)
    return selected or list(feature_columns)


def select_sybil_features(feature_columns: Sequence[str]) -> List[str]:
    selected = [col for col in feature_columns if col.startswith("sybil_") or col in {"rate_msgs_per_s", "window_id"}]
    return selected or list(feature_columns)


def select_integrity_features(feature_columns: Sequence[str]) -> List[str]:
    selected = [
        col
        for col in feature_columns
        if col.startswith("flag_") or col in {"proto_anom_count", "obu_flag_count", "obu_risk", "consistency_err"}
    ]
    return selected or list(feature_columns)


def build_feature_config(df: pd.DataFrame, *, window_size: int, seq_len: int) -> Dict[str, Any]:
    feature_columns = select_general_features(df)
    fill_values = (
        df[feature_columns]
        .median(numeric_only=True)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_dict()
    )
    head_features = {
        "general": list(feature_columns),
        "pos_speed": select_pos_speed_features(feature_columns),
        "replay_stale": list(feature_columns),
        "dos": select_dos_features(feature_columns),
        "sybil": select_sybil_features(feature_columns),
        "integrity": select_integrity_features(feature_columns),
    }
    binary_feature_columns = [col for col in feature_columns if is_binary_feature_series(df[col])]
    continuous_feature_columns = [col for col in feature_columns if col not in set(binary_feature_columns)]
    return {
        "window_size": int(window_size),
        "seq_len": int(seq_len),
        "feature_columns": list(feature_columns),
        "binary_feature_columns": list(binary_feature_columns),
        "continuous_feature_columns": list(continuous_feature_columns),
        "fill_values": {key: float(value) for key, value in fill_values.items()},
        "head_features": head_features,
        "head_order": list(HEAD_VECTOR_ORDER),
    }


def align_feature_matrix(df: pd.DataFrame, feature_config: Dict[str, Any]) -> pd.DataFrame:
    feature_columns = list(feature_config["feature_columns"])
    fill_values = feature_config.get("fill_values", {})
    aligned = pd.DataFrame(index=df.index)
    for col in feature_columns:
        if col in df.columns:
            aligned[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            aligned[col] = np.nan
        aligned[col] = aligned[col].fillna(float(fill_values.get(col, 0.0)))
    aligned = aligned.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return aligned.astype(float)


def fit_base_and_calibrator(
    estimator: Any,
    X: np.ndarray,
    y: pd.Series,
) -> Tuple[Any, Optional[CalibratedClassifierCV]]:
    model = clone(estimator)
    model.fit(X, y)

    calibrator: Optional[CalibratedClassifierCV] = None
    class_counts = pd.Series(y).value_counts()
    if len(class_counts) >= 2 and class_counts.min() >= 2:
        cv = int(min(3, class_counts.min()))
        if cv >= 2:
            calibrator = CalibratedClassifierCV(clone(estimator), method="isotonic", cv=cv)
            calibrator.fit(X, y)
    return model, calibrator


def predict_binary_proba(model: Any, calibrator: Optional[Any], X: np.ndarray) -> np.ndarray:
    if calibrator is not None:
        return calibrator.predict_proba(X)[:, 1]
    if model is not None:
        return model.predict_proba(X)[:, 1]
    return np.zeros(len(X), dtype=float)


def make_sequences_per_sender(
    X_df: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Xs: List[np.ndarray] = []
    ys: List[int] = []
    gs: List[str] = []
    idx_last: List[int] = []
    for sender, indices in X_df.groupby(groups, sort=False).groups.items():
        ordered_idx = np.array(sorted(indices))
        Xi = X_df.loc[ordered_idx].to_numpy(dtype=float)
        yi = y.loc[ordered_idx].to_numpy(dtype=int)
        if len(Xi) < seq_len:
            continue
        for start in range(0, len(Xi) - seq_len + 1):
            end = start + seq_len
            Xs.append(Xi[start:end])
            ys.append(int(yi[end - 1]))
            gs.append(str(sender))
            idx_last.append(int(ordered_idx[end - 1]))
    if not Xs:
        return (
            np.zeros((0, seq_len, X_df.shape[1]), dtype=float),
            np.zeros((0,), dtype=int),
            np.array([], dtype=object),
            np.array([], dtype=int),
        )
    return np.stack(Xs), np.array(ys), np.array(gs), np.array(idx_last)


def fit_replay_lstm(
    X_df_scaled: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    seq_len: int,
    epochs: int = 20,
) -> Tuple[Optional[tf.keras.Model], np.ndarray]:
    require_tensorflow("training the replay_stale LSTM head")
    X_seq, y_seq, g_seq, idx_last = make_sequences_per_sender(X_df_scaled, y, groups, seq_len=seq_len)
    if len(X_seq) == 0 or len(np.unique(y_seq)) < 2:
        return None, np.zeros(len(X_df_scaled), dtype=float)

    input_shape = (X_seq.shape[1], X_seq.shape[2])
    oof = pd.Series(0.0, index=X_df_scaled.index, dtype=float)

    class_counts = pd.Series(y_seq).value_counts()
    cv = int(min(3, class_counts.min(), len(pd.Series(g_seq).unique())))
    if cv >= 2:
        splitter = StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=42)
        for tr_idx, va_idx in splitter.split(X_seq, y_seq, g_seq):
            fold_model = build_lstm(input_shape)
            stopper = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
            fold_model.fit(
                X_seq[tr_idx],
                y_seq[tr_idx],
                epochs=epochs,
                batch_size=128,
                validation_data=(X_seq[va_idx], y_seq[va_idx]),
                callbacks=[stopper],
                verbose=0,
            )
            preds = fold_model.predict(X_seq[va_idx], verbose=0).ravel()
            oof.loc[idx_last[va_idx]] = preds

    model = build_lstm(input_shape)
    stopper = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    model.fit(
        X_seq,
        y_seq,
        epochs=epochs,
        batch_size=128,
        validation_split=0.2,
        callbacks=[stopper],
        verbose=0,
    )
    return model, oof.to_numpy(dtype=float)


def predict_replay_lstm(
    model: Optional[tf.keras.Model],
    X_df_scaled: pd.DataFrame,
    groups: pd.Series,
    *,
    seq_len: int,
) -> np.ndarray:
    if model is None:
        return np.zeros(len(X_df_scaled), dtype=float)
    y_dummy = pd.Series(np.zeros(len(X_df_scaled), dtype=int), index=X_df_scaled.index)
    X_seq, _, _, idx_last = make_sequences_per_sender(X_df_scaled, y_dummy, groups, seq_len=seq_len)
    if len(X_seq) == 0:
        return np.zeros(len(X_df_scaled), dtype=float)
    preds = pd.Series(0.0, index=X_df_scaled.index, dtype=float)
    preds.loc[idx_last] = model.predict(X_seq, verbose=0).ravel()
    return preds.reindex(X_df_scaled.index).to_numpy(dtype=float)


def _assert_group_support_for_split(y: pd.Series, groups: pd.Series, *, n_splits: int, context: str) -> None:
    if n_splits < 2:
        raise ValueError(f"{context} requires at least two folds.")
    grouped = pd.DataFrame({"group": groups.astype(str), "label": y.astype(int)}).groupby("group", sort=False)["label"].max()
    positive_groups = int(grouped.sum())
    negative_groups = int(len(grouped) - positive_groups)
    if positive_groups < n_splits or negative_groups < n_splits:
        raise ValueError(
            f"{context} requires at least {n_splits} malicious sender groups and {n_splits} benign sender groups."
        )


def split_sender_disjoint_train_val_test(
    df: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    random_state: int = 42,
) -> SenderDisjointSplits:
    group_values = groups.astype(str)
    _assert_group_support_for_split(y, group_values, n_splits=10, context="thesis train/val/test split")
    test_splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=random_state)
    trainval_idx, test_idx = next(test_splitter.split(df, y, group_values))

    trainval_df = df.iloc[trainval_idx]
    trainval_y = y.iloc[trainval_idx]
    trainval_groups = group_values.iloc[trainval_idx]
    _assert_group_support_for_split(trainval_y, trainval_groups, n_splits=9, context="thesis validation split")
    val_splitter = StratifiedGroupKFold(n_splits=9, shuffle=True, random_state=random_state + 1)
    train_rel_idx, val_rel_idx = next(val_splitter.split(trainval_df, trainval_y, trainval_groups))
    train_idx = trainval_idx[train_rel_idx]
    val_idx = trainval_idx[val_rel_idx]

    train_groups = set(group_values.iloc[train_idx].tolist())
    val_groups = set(group_values.iloc[val_idx].tolist())
    test_groups = set(group_values.iloc[test_idx].tolist())
    if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
        raise RuntimeError("Sender-disjoint split failed; sender overlap detected.")

    return SenderDisjointSplits(
        train_idx=np.asarray(train_idx, dtype=int),
        val_idx=np.asarray(val_idx, dtype=int),
        test_idx=np.asarray(test_idx, dtype=int),
    )


def build_meta_matrix(head_scores: Dict[str, np.ndarray]) -> np.ndarray:
    vectors = []
    for head_name in HEAD_VECTOR_ORDER:
        values = head_scores.get(head_name)
        if values is None:
            values = np.zeros(len(next(iter(head_scores.values()))), dtype=float)
        vectors.append(np.asarray(values, dtype=float))
    return np.vstack(vectors).T


def json_ready(value: Any) -> Any:
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def build_required_bundle_files() -> Dict[str, str]:
    return {
        "manifest": "manifest.json",
        "features": "features.json",
        "scaler": "scaler.joblib",
        "general_head": "general_head.joblib",
        "general_calibrator": "general_calibrator.joblib",
        "pos_speed_head": "pos_speed_head.joblib",
        "pos_speed_calibrator": "pos_speed_calibrator.joblib",
        "dos_head": "dos_head.joblib",
        "dos_calibrator": "dos_calibrator.joblib",
        "dos_iforest": "dos_iforest.joblib",
        "sybil_head": "sybil_head.joblib",
        "sybil_calibrator": "sybil_calibrator.joblib",
        "integrity_head": "integrity_head.joblib",
        "integrity_calibrator": "integrity_calibrator.joblib",
        "replay_lstm": "replay_lstm.keras",
        "replay_config": "replay_config.json",
        "meta_classifier": "meta_classifier.joblib",
        "trust_config": "trust_config.json",
        "obu_thresholds": "obu_thresholds.json",
        "training_report": "training_report.json",
    }


def is_complete_thesis_artifacts(artifacts: Dict[str, Any]) -> bool:
    head_models = artifacts.get("head_models", {})
    head_calibrators = artifacts.get("head_calibrators", {})
    if any(head_models.get(head_name) is None for head_name in STRICT_REQUIRED_MODEL_HEADS):
        return False
    if any(head_calibrators.get(head_name) is None for head_name in STRICT_REQUIRED_MODEL_HEADS):
        return False
    if artifacts.get("replay_model") is None:
        return False
    if artifacts.get("replay_config") is None:
        return False
    if artifacts.get("dos_iforest") is None:
        return False
    if artifacts.get("meta_classifier") is None:
        return False
    return True


@dataclass
class TrainingResult:
    output_dir: str
    manifest_path: str
    report_path: str
    enabled_heads: List[str]
    threshold: float
    report: Dict[str, Any]


@dataclass
class SenderDisjointSplits:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

    @property
    def train_size(self) -> int:
        return int(len(self.train_idx))

    @property
    def val_size(self) -> int:
        return int(len(self.val_idx))

    @property
    def test_size(self) -> int:
        return int(len(self.test_idx))


class AdaptiveTrustManager:
    """Maintains sender trust and converts it into an adaptive threshold."""

    def __init__(self, config: Dict[str, Any]):
        self.alpha0 = float(config.get("alpha0", 1.0))
        self.beta0 = float(config.get("beta0", 1.0))
        self.base_threshold = float(config.get("base_threshold", 0.370))
        self.sensitivity = float(config.get("sensitivity", 0.4))
        self.floor = float(config.get("floor", 0.35))
        self.ceil = float(config.get("ceil", 0.85))
        self.w_bad = float(config.get("w_bad", 1.0))
        self.w_good = float(config.get("w_good", 0.5))
        self.w_bad_minor = float(config.get("w_bad_minor", 0.2))
        self.state: Dict[str, Tuple[float, float]] = {}

    def _get_state(self, sender: str) -> Tuple[float, float]:
        return self.state.get(str(sender), (self.alpha0, self.beta0))

    def trust(self, sender: str) -> float:
        alpha, beta = self._get_state(sender)
        return alpha / max(alpha + beta, 1e-9)

    def update(self, sender: str, *, malicious: bool, weight: float) -> None:
        alpha, beta = self._get_state(sender)
        if malicious:
            beta += float(max(weight, 0.0))
        else:
            alpha += float(max(weight, 0.0))
        self.state[str(sender)] = (alpha, beta)

    def has_obu_flags(self, row: pd.Series) -> bool:
        return any(int(row.get(flag_col, 0)) == 1 for flag_col in OBU_FLAG_COLUMNS)

    def threshold(self, sender: str) -> float:
        trust = self.trust(sender)
        adjust = self.sensitivity * (trust - 0.5)
        return float(np.clip(self.base_threshold + adjust, self.floor, self.ceil))

    def update_after_decision(self, sender: str, *, decision: int, has_obu_flags: bool) -> None:
        if int(decision) == 1:
            self.update(sender, malicious=True, weight=self.w_bad)
            return
        self.update(sender, malicious=False, weight=self.w_good)
        if has_obu_flags:
            self.update(sender, malicious=True, weight=self.w_bad_minor)


class RSUMultiHeadTrainer:
    """Trains the thesis-aligned RSU multi-head models and saves a release bundle."""

    def __init__(
        self,
        *,
        train_family: str = "all",
        seq_len: int = DEFAULT_SEQ_LEN,
        window_size: int = DEFAULT_WINDOW_SIZE,
    ):
        if train_family not in TRAIN_FAMILY_CHOICES:
            raise ValueError(f"Unsupported train_family={train_family!r}")
        self.train_family = train_family
        self.seq_len = int(seq_len)
        self.window_size = int(window_size)

    @staticmethod
    def _general_estimator() -> lgb.LGBMClassifier:
        require_lightgbm("training the RSU general binary detector")
        return lgb.LGBMClassifier(
            objective="binary",
            class_weight="balanced",
            n_estimators=500,
            learning_rate=0.04,
            num_leaves=160,
            min_data_in_leaf=20,
            feature_fraction=0.85,
            random_state=42,
            min_gain_to_split=1e-12,
            verbosity=-1,
        )

    @staticmethod
    def _family_estimator() -> lgb.LGBMClassifier:
        require_lightgbm("training RSU family detectors")
        return lgb.LGBMClassifier(
            objective="binary",
            class_weight="balanced",
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=128,
            min_data_in_leaf=30,
            feature_fraction=0.90,
            random_state=42,
            min_gain_to_split=1e-12,
            verbosity=-1,
        )

    @staticmethod
    def _integrity_estimator() -> lgb.LGBMClassifier:
        require_lightgbm("training the RSU integrity detector")
        return lgb.LGBMClassifier(
            objective="binary",
            class_weight="balanced",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=96,
            min_data_in_leaf=20,
            feature_fraction=0.95,
            random_state=42,
            min_gain_to_split=1e-12,
            verbosity=-1,
        )

    def _requested_family_heads(self) -> List[str]:
        if self.train_family == "all":
            return ["pos_speed", "replay_stale", "dos", "sybil"]
        if self.train_family == "binary":
            return []
        return [self.train_family]

    def _active_head_names(self) -> List[str]:
        active = {"general", "integrity"}
        for family in self._requested_family_heads():
            active.add(family)
            if family == "dos":
                active.add("dos_iforest")
        return [head_name for head_name in HEAD_VECTOR_ORDER if head_name in active]

    def _head_target(self, df: pd.DataFrame, head_name: str) -> pd.Series:
        if head_name in {"general", "integrity"}:
            return df["label"].astype(int)
        if head_name == "dos_iforest":
            return family_labels(df, "dos")
        return family_labels(df, head_name)

    def _scaled_features(
        self,
        df: pd.DataFrame,
        feature_config: Dict[str, Any],
        scaler: Optional[StandardScaler] = None,
    ) -> Tuple[Optional[StandardScaler], pd.DataFrame]:
        aligned = align_feature_matrix(df, feature_config)
        if scaler is None:
            scaler = fit_feature_scaler(aligned, feature_config)
        scaled = transform_feature_matrix(aligned, feature_config, scaler)
        return scaler, scaled

    def _fit_binary_head(
        self,
        *,
        estimator: lgb.LGBMClassifier,
        head_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Tuple[Any, CalibratedClassifierCV]:
        model, calibrator = fit_base_and_calibrator(estimator, X_train.to_numpy(dtype=float), y_train)
        if calibrator is None:
            raise ValueError(f"{head_name} requires a calibration model, but calibration could not be trained.")
        return model, calibrator

    def _fit_single_head_artifacts(
        self,
        fit_df: pd.DataFrame,
        feature_config: Dict[str, Any],
        *,
        active_heads: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        requested_heads = (
            [head_name for head_name in active_heads]
            if active_heads is not None
            else self._active_head_names()
        )
        scaler, X_scaled = self._scaled_features(fit_df, feature_config)
        artifacts: Dict[str, Any] = {
            "scaler": scaler,
            "feature_config": feature_config,
            "head_models": {},
            "head_calibrators": {},
            "enabled_heads": [],
            "replay_model": None,
            "replay_config": None,
            "dos_iforest": None,
            "meta_classifier": None,
        }

        y_binary = fit_df["label"].astype(int)
        enabled_heads: set[str] = set()
        if "general" in requested_heads:
            general_model, general_calibrator = self._fit_binary_head(
                estimator=self._general_estimator(),
                head_name="general",
                X_train=X_scaled[feature_config["head_features"]["general"]],
                y_train=y_binary,
            )
            artifacts["head_models"]["general"] = general_model
            artifacts["head_calibrators"]["general"] = general_calibrator
            enabled_heads.add("general")

        if "integrity" in requested_heads:
            integrity_model, integrity_calibrator = self._fit_binary_head(
                estimator=self._integrity_estimator(),
                head_name="integrity",
                X_train=X_scaled[feature_config["head_features"]["integrity"]],
                y_train=y_binary,
            )
            artifacts["head_models"]["integrity"] = integrity_model
            artifacts["head_calibrators"]["integrity"] = integrity_calibrator
            enabled_heads.add("integrity")

        family_heads = ["pos_speed", "replay_stale", "dos", "sybil"]
        for family in family_heads:
            if family not in requested_heads and not (family == "dos" and "dos_iforest" in requested_heads):
                continue
            y_family = family_labels(fit_df, family)
            if len(np.unique(y_family)) < 2:
                raise ValueError(f"{family} cannot be trained because the requested fit split lacks both classes.")

            if family == "replay_stale" and family in requested_heads:
                replay_model, _ = fit_replay_lstm(
                    X_scaled[feature_config["head_features"]["replay_stale"]],
                    y_family,
                    fit_df["sender_pseudo"].astype(str),
                    seq_len=self.seq_len,
                )
                if replay_model is None:
                    raise ValueError("replay_stale could not be trained; the sequence split did not produce a valid LSTM model.")
                artifacts["replay_model"] = replay_model
                artifacts["replay_config"] = {
                    "seq_len": self.seq_len,
                    "feature_columns": feature_config["head_features"]["replay_stale"],
                }
                enabled_heads.add("replay_stale")
                continue

            selected_features = feature_config["head_features"][family]
            if family in requested_heads:
                model, calibrator = self._fit_binary_head(
                    estimator=self._family_estimator(),
                    head_name=family,
                    X_train=X_scaled[selected_features],
                    y_train=y_family,
                )
                artifacts["head_models"][family] = model
                artifacts["head_calibrators"][family] = calibrator
                enabled_heads.add(family)

            if family == "dos":
                normal_mask = y_family.to_numpy(dtype=int) == 0
                if int(normal_mask.sum()) <= 10:
                    raise ValueError("dos_iforest requires more benign DoS-negative samples in the fit split.")
                iforest = IsolationForest(
                    n_estimators=300,
                    contamination=0.02,
                    random_state=42,
                    n_jobs=-1,
                )
                X_dos = X_scaled[selected_features].to_numpy(dtype=float)
                iforest.fit(X_dos[normal_mask])
                scores = -iforest.score_samples(X_dos)
                lo, hi = np.percentile(scores, [1, 99])
                if hi <= lo:
                    hi = lo + 1e-6
                artifacts["dos_iforest"] = {
                    "model": iforest,
                    "score_min": float(lo),
                    "score_max": float(hi),
                    "feature_names": selected_features,
                }
                if "dos_iforest" in requested_heads:
                    enabled_heads.add("dos_iforest")

        artifacts["enabled_heads"] = [head_name for head_name in HEAD_VECTOR_ORDER if head_name in enabled_heads]
        return artifacts

    def _predict_single_head(
        self,
        head_name: str,
        artifacts: Dict[str, Any],
        score_df: pd.DataFrame,
        feature_config: Dict[str, Any],
    ) -> np.ndarray:
        scaler = artifacts["scaler"]
        _, X_scaled = self._scaled_features(score_df, feature_config, scaler=scaler)

        if head_name == "general":
            return predict_binary_proba(
                artifacts["head_models"]["general"],
                artifacts["head_calibrators"]["general"],
                X_scaled[feature_config["head_features"]["general"]].to_numpy(dtype=float),
            )
        if head_name == "integrity":
            return predict_binary_proba(
                artifacts["head_models"]["integrity"],
                artifacts["head_calibrators"]["integrity"],
                X_scaled[feature_config["head_features"]["integrity"]].to_numpy(dtype=float),
            )
        if head_name == "replay_stale":
            return predict_replay_lstm(
                artifacts["replay_model"],
                X_scaled[feature_config["head_features"]["replay_stale"]],
                score_df["sender_pseudo"].astype(str),
                seq_len=self.seq_len,
            )
        if head_name == "dos_iforest":
            bundle = artifacts["dos_iforest"]
            X_dos = X_scaled[bundle["feature_names"]].to_numpy(dtype=float)
            raw_score = -bundle["model"].score_samples(X_dos)
            lo = float(bundle["score_min"])
            hi = float(bundle["score_max"])
            return np.clip((raw_score - lo) / (hi - lo + 1e-9), 0.0, 1.0)

        selected_features = feature_config["head_features"][head_name]
        return predict_binary_proba(
            artifacts["head_models"][head_name],
            artifacts["head_calibrators"][head_name],
            X_scaled[selected_features].to_numpy(dtype=float),
        )

    def _oof_scores_for_head(
        self,
        train_df: pd.DataFrame,
        feature_config: Dict[str, Any],
        head_name: str,
    ) -> np.ndarray:
        y_head = self._head_target(train_df, head_name)
        groups = train_df["sender_pseudo"].astype(str)
        _assert_group_support_for_split(y_head, groups, n_splits=5, context=f"{head_name} OOF stacking")
        splitter = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=100 + HEAD_VECTOR_ORDER.index(head_name),
        )
        oof_scores = pd.Series(0.0, index=train_df.index, dtype=float)
        for train_idx, valid_idx in splitter.split(train_df, y_head, groups):
            fold_train = train_df.iloc[train_idx].copy()
            fold_valid = train_df.iloc[valid_idx].copy()
            fold_artifacts = self._fit_single_head_artifacts(
                fold_train,
                feature_config,
                active_heads=[head_name],
            )
            oof_scores.loc[fold_valid.index] = self._predict_single_head(
                head_name,
                fold_artifacts,
                fold_valid,
                feature_config,
            )
        return oof_scores.reindex(train_df.index).to_numpy(dtype=float)

    def _fit_meta_classifier(
        self,
        fit_df: pd.DataFrame,
        feature_config: Dict[str, Any],
    ) -> Tuple[LogisticRegression, Dict[str, np.ndarray]]:
        head_scores = {head_name: np.zeros(len(fit_df), dtype=float) for head_name in HEAD_VECTOR_ORDER}
        for head_name in self._active_head_names():
            head_scores[head_name] = self._oof_scores_for_head(fit_df, feature_config, head_name)
        meta = LogisticRegression(class_weight="balanced", max_iter=400, random_state=42, C=1.0)
        meta.fit(build_meta_matrix(head_scores), fit_df["label"].astype(int))
        return meta, head_scores

    def fit(self, raw_df: pd.DataFrame, *, output_dir: str, source_name: str = "dataset") -> TrainingResult:
        if "label" not in raw_df.columns:
            raise ValueError("Training data must include a 'label' column.")

        prepared_df, obu_thresholds = prepare_thesis_dataframe(
            raw_df,
            window_size=self.window_size,
            obu_thresholds=thesis_obu_thresholds(),
        )
        if prepared_df.empty:
            raise ValueError("Prepared training dataframe is empty.")

        y_all = prepared_df["label"].astype(int)
        groups = prepared_df["sender_pseudo"].astype(str)
        splits = split_sender_disjoint_train_val_test(prepared_df, y_all, groups)
        train_df = prepared_df.iloc[splits.train_idx].copy()
        val_df = prepared_df.iloc[splits.val_idx].copy()
        test_df = prepared_df.iloc[splits.test_idx].copy()

        feature_config = build_feature_config(
            train_df,
            window_size=self.window_size,
            seq_len=self.seq_len,
        )

        meta_train, oof_scores_train = self._fit_meta_classifier(train_df, feature_config)
        train_only_artifacts = self._fit_single_head_artifacts(train_df, feature_config)
        train_only_artifacts["meta_classifier"] = meta_train

        eval_runtime = RSUMultiHeadRuntime.from_trained_artifacts(
            output_dir=output_dir,
            feature_config=feature_config,
            obu_thresholds=obu_thresholds,
            trust_config={},
            artifacts=train_only_artifacts,
        )
        eval_runtime.meta_classifier = meta_train
        scored_val = eval_runtime._score_heads(val_df, require_complete=False)
        if "p_final" not in scored_val.columns:
            raise RuntimeError("Validation scoring failed to produce p_final for threshold selection.")
        threshold = best_f1_threshold(val_df["label"].astype(int), scored_val["p_final"].to_numpy(dtype=float), default=0.370)

        trust_config = {
            "alpha0": 1.0,
            "beta0": 1.0,
            "base_threshold": float(threshold),
            "sensitivity": 0.4,
            "floor": 0.35,
            "ceil": 0.85,
            "w_bad": 1.0,
            "w_good": 0.5,
            "w_bad_minor": 0.2,
        }

        final_fit_df = pd.concat([train_df, val_df], axis=0).sort_index(kind="mergesort")
        final_feature_config = build_feature_config(
            final_fit_df,
            window_size=self.window_size,
            seq_len=self.seq_len,
        )
        final_artifacts = self._fit_single_head_artifacts(final_fit_df, final_feature_config)
        final_meta, _ = self._fit_meta_classifier(final_fit_df, final_feature_config)
        final_artifacts["meta_classifier"] = final_meta

        if self.train_family == "all" and not is_complete_thesis_artifacts(final_artifacts):
            raise ValueError("Refusing to save an incomplete thesis-grade multi-head bundle.")

        runtime = RSUMultiHeadRuntime.from_trained_artifacts(
            output_dir=output_dir,
            feature_config=final_feature_config,
            obu_thresholds=obu_thresholds,
            trust_config=trust_config,
            artifacts=final_artifacts,
        )
        runtime.meta_classifier = final_meta
        scored_test = runtime.score_dataframe(test_df)

        y_test = test_df["label"].astype(int).to_numpy()
        p_final_test = scored_test["p_final"].to_numpy(dtype=float)
        fixed_pred = (p_final_test >= threshold).astype(int)
        adaptive_pred = scored_test["final_decision"].to_numpy(dtype=int)
        roc_auc = float(roc_auc_score(y_test, p_final_test)) if len(np.unique(y_test)) >= 2 else None

        enabled_heads = final_artifacts["enabled_heads"]
        report = {
            "artifact_family": ARTIFACT_FAMILY,
            "artifact_version": ARTIFACT_VERSION,
            "source_name": source_name,
            "rows": int(len(prepared_df)),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "split_protocol": "sender_disjoint_80_10_10",
            "meta_protocol": "5fold_stratified_group_oof",
            "feature_count": int(len(final_feature_config["feature_columns"])),
            "binary_feature_count": int(len(final_feature_config["binary_feature_columns"])),
            "continuous_feature_count": int(len(final_feature_config["continuous_feature_columns"])),
            "enabled_heads": enabled_heads,
            "used_family_heads": self._requested_family_heads(),
            "threshold": float(threshold),
            "roc_auc": roc_auc,
            "fixed_confusion_matrix": json_ready(confusion_matrix(y_test, fixed_pred)),
            "adaptive_confusion_matrix": json_ready(confusion_matrix(y_test, adaptive_pred)),
            "fixed_classification_report": json_ready(
                classification_report(y_test, fixed_pred, digits=4, output_dict=True, zero_division=0)
            ),
            "adaptive_classification_report": json_ready(
                classification_report(y_test, adaptive_pred, digits=4, output_dict=True, zero_division=0)
            ),
            "oof_meta_rows": int(len(next(iter(oof_scores_train.values()))) if oof_scores_train else 0),
        }

        runtime.save_release_bundle(output_dir=output_dir, training_report=report, source_name=source_name)
        report_path = str(Path(output_dir) / "training_report.json")
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(json_ready(report), handle, indent=2)

        return TrainingResult(
            output_dir=output_dir,
            manifest_path=str(Path(output_dir) / "manifest.json"),
            report_path=report_path,
            enabled_heads=enabled_heads,
            threshold=float(threshold),
            report=report,
        )

    def fit_csv(self, csv_path: str, *, output_dir: str) -> TrainingResult:
        raw_df = pd.read_csv(csv_path, low_memory=False)
        return self.fit(raw_df, output_dir=output_dir, source_name=csv_path)


class RSUMultiHeadRuntime:
    """Loads the thesis-aligned release bundle and runs unified RSU inference."""

    def __init__(
        self,
        *,
        models_dir: Optional[str] = None,
        scaler: Optional[StandardScaler] = None,
        feature_config: Optional[Dict[str, Any]] = None,
        obu_thresholds: Optional[Dict[str, float]] = None,
        trust_config: Optional[Dict[str, Any]] = None,
        head_models: Optional[Dict[str, Any]] = None,
        head_calibrators: Optional[Dict[str, Any]] = None,
        replay_model: Optional[tf.keras.Model] = None,
        replay_config: Optional[Dict[str, Any]] = None,
        dos_iforest: Optional[Dict[str, Any]] = None,
        meta_classifier: Optional[LogisticRegression] = None,
        manifest: Optional[Dict[str, Any]] = None,
    ):
        self.models_dir = models_dir
        self.scaler = scaler
        self.feature_config = feature_config or {}
        self.obu_thresholds = obu_thresholds or {}
        self.trust_config = trust_config or {}
        self.head_models = head_models or {}
        self.head_calibrators = head_calibrators or {}
        self.replay_model = replay_model
        self.replay_config = replay_config or {}
        self.dos_iforest = dos_iforest
        self.meta_classifier = meta_classifier
        self.manifest = manifest or {}

    @classmethod
    def from_trained_artifacts(
        cls,
        *,
        output_dir: str,
        feature_config: Dict[str, Any],
        obu_thresholds: Dict[str, float],
        trust_config: Dict[str, Any],
        artifacts: Dict[str, Any],
    ) -> "RSUMultiHeadRuntime":
        return cls(
            models_dir=output_dir,
            scaler=artifacts["scaler"],
            feature_config=feature_config,
            obu_thresholds=obu_thresholds,
            trust_config=trust_config,
            head_models=artifacts["head_models"],
            head_calibrators=artifacts["head_calibrators"],
            replay_model=artifacts.get("replay_model"),
            replay_config=artifacts.get("replay_config"),
            dos_iforest=artifacts.get("dos_iforest"),
            meta_classifier=artifacts.get("meta_classifier"),
            manifest={
                "artifact_family": ARTIFACT_FAMILY,
                "artifact_version": ARTIFACT_VERSION,
                "thesis_grade_complete": bool(is_complete_thesis_artifacts(artifacts)),
            },
        )

    def _runtime_artifacts_snapshot(self) -> Dict[str, Any]:
        return {
            "head_models": self.head_models,
            "head_calibrators": self.head_calibrators,
            "replay_model": self.replay_model,
            "replay_config": self.replay_config,
            "dos_iforest": self.dos_iforest,
            "meta_classifier": self.meta_classifier,
        }

    @staticmethod
    def _verify_complete_manifest(model_path: Path, manifest: Dict[str, Any]) -> None:
        if not bool(manifest.get("thesis_grade_complete", False)):
            raise ValueError(f"{model_path} is not a complete thesis-grade release bundle.")
        required_files = build_required_bundle_files()
        missing = [rel_path for rel_path in required_files.values() if not (model_path / rel_path).exists()]
        if missing:
            raise FileNotFoundError(f"Incomplete thesis-grade bundle in {model_path}: missing {missing}")

    @classmethod
    def load(cls, models_dir: str) -> "RSUMultiHeadRuntime":
        model_path = Path(models_dir)
        manifest_path = model_path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found in {models_dir}")
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("artifact_family") != ARTIFACT_FAMILY:
            raise ValueError(f"{models_dir} is not a {ARTIFACT_FAMILY} release bundle.")
        cls._verify_complete_manifest(model_path, manifest)

        with open(model_path / "features.json", "r", encoding="utf-8") as handle:
            feature_config = json.load(handle)
        with open(model_path / "obu_thresholds.json", "r", encoding="utf-8") as handle:
            obu_thresholds = json.load(handle)
        with open(model_path / "trust_config.json", "r", encoding="utf-8") as handle:
            trust_config = json.load(handle)

        scaler = _load_joblib_compat(model_path / "scaler.joblib")
        head_models = {}
        head_calibrators = {}

        for head in ("general", "pos_speed", "dos", "sybil", "integrity"):
            head_path = model_path / f"{head}_head.joblib"
            calib_path = model_path / f"{head}_calibrator.joblib"
            if head_path.exists():
                head_models[head] = _load_joblib_compat(head_path)
            if calib_path.exists():
                head_calibrators[head] = _load_joblib_compat(calib_path)

        replay_model = None
        if (model_path / "replay_lstm.keras").exists():
            require_tensorflow(f"loading replay_lstm.keras from {models_dir}")
            replay_model = load_model(model_path / "replay_lstm.keras")
        replay_config = {}
        if (model_path / "replay_config.json").exists():
            with open(model_path / "replay_config.json", "r", encoding="utf-8") as handle:
                replay_config = json.load(handle)

        dos_iforest = None
        if (model_path / "dos_iforest.joblib").exists():
            dos_iforest = _load_joblib_compat(model_path / "dos_iforest.joblib")

        meta_classifier = _load_joblib_compat(model_path / "meta_classifier.joblib")
        return cls(
            models_dir=models_dir,
            scaler=scaler,
            feature_config=feature_config,
            obu_thresholds=obu_thresholds,
            trust_config=trust_config,
            head_models=head_models,
            head_calibrators=head_calibrators,
            replay_model=replay_model,
            replay_config=replay_config,
            dos_iforest=dos_iforest,
            meta_classifier=meta_classifier,
            manifest=manifest,
        )

    def save_release_bundle(self, *, output_dir: str, training_report: Dict[str, Any], source_name: str) -> None:
        output_path = Path(ensure_dir(output_dir))
        required_files = build_required_bundle_files()

        with open(output_path / "features.json", "w", encoding="utf-8") as handle:
            json.dump(json_ready(self.feature_config), handle, indent=2)
        with open(output_path / "obu_thresholds.json", "w", encoding="utf-8") as handle:
            json.dump(json_ready(self.obu_thresholds), handle, indent=2)
        with open(output_path / "trust_config.json", "w", encoding="utf-8") as handle:
            json.dump(json_ready(self.trust_config), handle, indent=2)

        joblib.dump(self.scaler, output_path / "scaler.joblib")

        for head_name, model in self.head_models.items():
            joblib.dump(model, output_path / f"{head_name}_head.joblib")
        for head_name, calibrator in self.head_calibrators.items():
            if calibrator is not None:
                joblib.dump(calibrator, output_path / f"{head_name}_calibrator.joblib")

        if self.dos_iforest is not None:
            joblib.dump(self.dos_iforest, output_path / "dos_iforest.joblib")
        if self.replay_model is not None:
            self.replay_model.save(output_path / "replay_lstm.keras")
        if self.replay_config:
            with open(output_path / "replay_config.json", "w", encoding="utf-8") as handle:
                json.dump(json_ready(self.replay_config), handle, indent=2)
        if self.meta_classifier is not None:
            joblib.dump(self.meta_classifier, output_path / "meta_classifier.joblib")

        with open(output_path / "training_report.json", "w", encoding="utf-8") as handle:
            json.dump(json_ready(training_report), handle, indent=2)

        thesis_grade_complete = bool(is_complete_thesis_artifacts(self._runtime_artifacts_snapshot()))

        manifest = {
            "artifact_family": ARTIFACT_FAMILY,
            "artifact_version": ARTIFACT_VERSION,
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_name": source_name,
            "feature_count": int(len(self.feature_config.get("feature_columns", []))),
            "thesis_grade_complete": thesis_grade_complete,
            "enabled_heads": (
                sorted(self.head_models.keys())
                + (["replay_stale"] if self.replay_model is not None else [])
                + (["dos_iforest"] if self.dos_iforest is not None else [])
            ),
            "files": required_files,
            "training_report_summary": {
                "threshold": training_report.get("threshold"),
                "roc_auc": training_report.get("roc_auc"),
            },
        }
        with open(output_path / "manifest.json", "w", encoding="utf-8") as handle:
            json.dump(json_ready(manifest), handle, indent=2)
        self.manifest = manifest

    def _score_heads(self, prepared_df: pd.DataFrame, *, require_complete: bool = True) -> pd.DataFrame:
        if require_complete and not bool(self.manifest.get("thesis_grade_complete", False)):
            raise RuntimeError("Refusing to score with an incomplete thesis-grade bundle.")
        X = align_feature_matrix(prepared_df, self.feature_config)
        X_scaled = transform_feature_matrix(X, self.feature_config, self.scaler)

        scores: Dict[str, np.ndarray] = {}
        if require_complete and self.head_models.get("general") is None:
            raise RuntimeError("general head is missing from the loaded thesis-grade bundle.")
        scores["general"] = predict_binary_proba(
            self.head_models.get("general"),
            self.head_calibrators.get("general"),
            X_scaled[self.feature_config["head_features"]["general"]].to_numpy(dtype=float),
        )

        if require_complete and self.head_models.get("pos_speed") is None:
            raise RuntimeError("pos_speed head is missing from the loaded thesis-grade bundle.")
        if self.head_models.get("pos_speed") is not None or self.head_calibrators.get("pos_speed") is not None:
            pos_features = self.feature_config["head_features"]["pos_speed"]
            scores["pos_speed"] = predict_binary_proba(
                self.head_models.get("pos_speed"),
                self.head_calibrators.get("pos_speed"),
                X_scaled[pos_features].to_numpy(dtype=float),
            )
        else:
            scores["pos_speed"] = np.zeros(len(prepared_df), dtype=float)

        if require_complete and self.replay_model is None:
            raise RuntimeError("replay_stale LSTM is missing from the loaded thesis-grade bundle.")
        if self.replay_model is not None:
            scores["replay_stale"] = predict_replay_lstm(
                self.replay_model,
                X_scaled[self.feature_config["head_features"]["replay_stale"]],
                prepared_df["sender_pseudo"].astype(str),
                seq_len=int(self.replay_config.get("seq_len", self.feature_config.get("seq_len", DEFAULT_SEQ_LEN))),
            )
        else:
            scores["replay_stale"] = np.zeros(len(prepared_df), dtype=float)

        if require_complete and self.head_models.get("dos") is None:
            raise RuntimeError("dos head is missing from the loaded thesis-grade bundle.")
        if self.head_models.get("dos") is not None or self.head_calibrators.get("dos") is not None:
            dos_features = self.feature_config["head_features"]["dos"]
            X_dos = X_scaled[dos_features].to_numpy(dtype=float)
            scores["dos"] = predict_binary_proba(
                self.head_models.get("dos"),
                self.head_calibrators.get("dos"),
                X_dos,
            )
            if require_complete and self.dos_iforest is None:
                raise RuntimeError("dos_iforest is missing from the loaded thesis-grade bundle.")
            if self.dos_iforest is not None:
                raw_score = -self.dos_iforest["model"].score_samples(X_dos)
                lo = float(self.dos_iforest["score_min"])
                hi = float(self.dos_iforest["score_max"])
                scores["dos_iforest"] = np.clip((raw_score - lo) / (hi - lo + 1e-9), 0.0, 1.0)
            else:
                scores["dos_iforest"] = np.zeros(len(prepared_df), dtype=float)
        else:
            scores["dos"] = np.zeros(len(prepared_df), dtype=float)
            scores["dos_iforest"] = np.zeros(len(prepared_df), dtype=float)

        if require_complete and self.head_models.get("sybil") is None:
            raise RuntimeError("sybil head is missing from the loaded thesis-grade bundle.")
        if self.head_models.get("sybil") is not None or self.head_calibrators.get("sybil") is not None:
            sybil_features = self.feature_config["head_features"]["sybil"]
            scores["sybil"] = predict_binary_proba(
                self.head_models.get("sybil"),
                self.head_calibrators.get("sybil"),
                X_scaled[sybil_features].to_numpy(dtype=float),
            )
        else:
            scores["sybil"] = np.zeros(len(prepared_df), dtype=float)

        if require_complete and self.head_models.get("integrity") is None:
            raise RuntimeError("integrity head is missing from the loaded thesis-grade bundle.")
        integrity_features = self.feature_config["head_features"]["integrity"]
        scores["integrity"] = predict_binary_proba(
            self.head_models.get("integrity"),
            self.head_calibrators.get("integrity"),
            X_scaled[integrity_features].to_numpy(dtype=float),
        )

        out = prepared_df.copy()
        for head_name in HEAD_VECTOR_ORDER:
            out[f"p_{head_name}"] = scores.get(head_name, np.zeros(len(prepared_df), dtype=float))
        if self.meta_classifier is not None:
            meta_input = build_meta_matrix(scores)
            out["p_final"] = self.meta_classifier.predict_proba(meta_input)[:, 1]
        return out

    def score_dataframe(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        prepared_df, _ = prepare_thesis_dataframe(
            raw_df,
            window_size=int(self.feature_config.get("window_size", DEFAULT_WINDOW_SIZE)),
            obu_thresholds=self.obu_thresholds or thesis_obu_thresholds(),
        )
        scored = self._score_heads(prepared_df, require_complete=True)
        if "p_final" not in scored.columns:
            raise RuntimeError("Meta classifier is not loaded. The release bundle is incomplete.")
        trust = AdaptiveTrustManager(self.trust_config)
        scored = scored.sort_values(["t_curr", "sender_pseudo", "row_id"], kind="mergesort").reset_index(drop=True)

        trust_before = []
        trust_after = []
        adaptive_thresholds = []
        final_decisions = []

        for _, row in scored.iterrows():
            sender = str(row["sender_pseudo"])
            before = trust.trust(sender)
            threshold = trust.threshold(sender)
            decision = int(float(row["p_final"]) >= threshold)
            trust.update_after_decision(sender, decision=decision, has_obu_flags=trust.has_obu_flags(row))
            after = trust.trust(sender)

            trust_before.append(before)
            adaptive_thresholds.append(threshold)
            final_decisions.append(decision)
            trust_after.append(after)

        scored["trust_sender_before"] = trust_before
        scored["adaptive_threshold"] = adaptive_thresholds
        scored["final_decision"] = final_decisions
        scored["trust_sender_after"] = trust_after

        # Backward-compatible aliases for older verification tooling.
        scored["obu_risk"] = scored["obu_risk"].astype(float)
        scored["obu_anom"] = scored["obu_anom"].astype(int)
        scored["rsu_score"] = scored["p_final"].astype(float)
        scored["rsu_anom"] = scored["final_decision"].astype(int)
        scored["fused_risk"] = scored["p_final"].astype(float)
        scored["fused_anom"] = scored["final_decision"].astype(int)
        scored["runtime_mode"] = ARTIFACT_FAMILY

        keep_order = [
            "row_id",
            "row_key",
            "receiver_pseudo",
            "sender_pseudo",
            "t_curr",
            "dt",
            "label",
            "attack_id",
            *OBU_FLAG_COLUMNS,
            "obu_flag_count",
            "obu_risk",
            "obu_anom",
            "p_general",
            "p_pos_speed",
            "p_replay_stale",
            "p_dos",
            "p_dos_iforest",
            "p_sybil",
            "p_integrity",
            "p_final",
            "adaptive_threshold",
            "trust_sender_before",
            "trust_sender_after",
            "final_decision",
            "rsu_score",
            "rsu_anom",
            "fused_risk",
            "fused_anom",
            "runtime_mode",
        ]
        available = [col for col in keep_order if col in scored.columns]
        return scored[available].copy()

    def detect_csv(self, input_csv: str, output_csv: str) -> pd.DataFrame:
        raw_df = pd.read_csv(input_csv, low_memory=False)
        scored = self.score_dataframe(raw_df)
        scored.to_csv(output_csv, index=False)
        return scored

    def detect_raw_directory(self, root_dir: str, output_csv: str) -> pd.DataFrame:
        raw_df = load_raw_bsm_directory(root_dir)
        scored = self.score_dataframe(raw_df)
        scored.to_csv(output_csv, index=False)
        return scored

    def detect_live_source(
        self,
        *,
        source_path: str,
        output_csv: str,
        source_kind: str,
        poll_interval: float = 2.0,
        max_polls: Optional[int] = None,
        once: bool = False,
    ) -> pd.DataFrame:
        emitted_keys: set[str] = set()
        accumulated_output = pd.DataFrame()
        polls = 0

        while True:
            if source_kind == "raw-dir":
                raw_df = load_raw_bsm_directory(source_path)
            elif source_kind == "csv":
                raw_df = pd.read_csv(source_path, low_memory=False) if Path(source_path).exists() else pd.DataFrame()
            else:
                raise ValueError(f"Unsupported live source_kind={source_kind!r}")

            if not raw_df.empty:
                scored = self.score_dataframe(raw_df)
                new_rows = scored[~scored["row_key"].astype(str).isin(emitted_keys)].copy()
                if not new_rows.empty:
                    emitted_keys.update(new_rows["row_key"].astype(str).tolist())
                    write_header = not Path(output_csv).exists()
                    new_rows.to_csv(output_csv, mode="a", header=write_header, index=False)
                    accumulated_output = pd.concat([accumulated_output, new_rows], ignore_index=True)

            polls += 1
            if once:
                return accumulated_output
            if max_polls is not None and polls >= max_polls:
                return accumulated_output
            time.sleep(float(poll_interval))


def is_multi_head_release(models_dir: str) -> bool:
    manifest_path = Path(models_dir) / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        return manifest.get("artifact_family") == ARTIFACT_FAMILY
    except Exception:
        return False


def has_legacy_simple_runtime(models_dir: str) -> bool:
    return (Path(models_dir) / "fusion_config_final_v2.json").exists()


def is_legacy_family_bundle(models_dir: str) -> bool:
    model_path = Path(models_dir)
    if not model_path.is_dir():
        return False
    required = [
        model_path / "model_meta.json",
        model_path / "preproc.joblib",
        model_path / "bin_calib.joblib",
        model_path / "meta.joblib",
    ]
    if not all(path.exists() for path in required):
        return False
    meta = _try_json(model_path / "model_meta.json")
    return bool(meta.get("family"))


def latest_archived_family_bundles(models_dir: str) -> Dict[str, str]:
    root_path = Path(models_dir)
    if not root_path.is_dir():
        return {}
    candidates: Dict[str, List[Tuple[str, str]]] = {family: [] for family in ARCHIVED_FAMILY_ORDER}
    scan_dirs = [root_path, *sorted(path for path in root_path.rglob("*") if path.is_dir())]
    for path in scan_dirs:
        if not is_legacy_family_bundle(str(path)):
            continue
        meta = _try_json(path / "model_meta.json")
        family = str(meta.get("family") or "")
        if family not in candidates:
            continue
        created = str(meta.get("created") or path.name)
        candidates[family].append((created, str(path)))
    selected: Dict[str, str] = {}
    for family, rows in candidates.items():
        if not rows:
            continue
        preferred = PREFERRED_ARCHIVED_FAMILY_REVISIONS.get(family)
        if preferred:
            preferred_rows = [row for row in rows if Path(row[1]).name == preferred]
            if preferred_rows:
                selected[family] = sorted(preferred_rows, key=lambda item: (item[0], item[1]))[-1][1]
                continue
        selected[family] = sorted(rows, key=lambda item: (item[0], item[1]))[-1][1]
    return selected


def is_archived_family_ensemble(models_dir: str) -> bool:
    bundles = latest_archived_family_bundles(models_dir)
    return all(family in bundles for family in ARCHIVED_FAMILY_ORDER)


def is_scenario_lgbm_bundle(models_dir: str) -> bool:
    model_path = Path(models_dir)
    manifest_path = model_path / "manifest.json"
    if not manifest_path.exists() or not (model_path / "scenario_lgbm.joblib").exists():
        return False
    try:
        manifest = _load_json(manifest_path)
    except Exception:
        return False
    return manifest.get("artifact_family") == SCENARIO_LGBM_ARTIFACT


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _try_json(path: Path) -> Dict[str, Any]:
    try:
        return _load_json(path)
    except Exception:
        return {}


def discover_model_bundles(root_dir: str) -> List[Dict[str, Any]]:
    root_path = Path(root_dir)
    if not root_path.exists():
        return []

    scan_dirs: List[Path] = []
    if root_path.is_dir():
        scan_dirs.append(root_path)
        scan_dirs.extend(sorted(path for path in root_path.rglob("*") if path.is_dir()))
    else:
        scan_dirs.append(root_path.parent)

    found: Dict[str, Dict[str, Any]] = {}
    for path in scan_dirs:
        key = str(path.resolve())
        if key in found:
            continue
        if is_multi_head_release(str(path)):
            manifest = _try_json(path / "manifest.json")
            found[key] = {
                "path": str(path),
                "type": manifest.get("artifact_family", ARTIFACT_FAMILY),
                "version": manifest.get("artifact_version"),
                "feature_count": manifest.get("feature_count"),
                "family": None,
                "created": manifest.get("built_at"),
                "score_runtime": "rsu_multi_head_runtime",
            }
            continue
        if is_scenario_lgbm_bundle(str(path)):
            manifest = _try_json(path / "manifest.json")
            found[key] = {
                "path": str(path),
                "type": SCENARIO_LGBM_ARTIFACT,
                "version": manifest.get("artifact_version"),
                "feature_count": manifest.get("feature_count"),
                "family": manifest.get("trained_family"),
                "created": manifest.get("built_at"),
                "score_runtime": "scenario_lgbm_runtime",
            }
            continue
        if has_legacy_simple_runtime(str(path)):
            found[key] = {
                "path": str(path),
                "type": "legacy_simple_runtime",
                "version": "v2",
                "feature_count": None,
                "family": None,
                "created": None,
                "score_runtime": "legacy_simple_runtime",
            }
            continue
        if is_archived_family_ensemble(str(path)):
            bundles = latest_archived_family_bundles(str(path))
            found[key] = {
                "path": str(path),
                "type": ARCHIVED_FAMILY_ENSEMBLE_ARTIFACT,
                "version": "latest-per-family",
                "feature_count": None,
                "family": "all",
                "created": ",".join(
                    Path(bundles[family]).name for family in ARCHIVED_FAMILY_ORDER if family in bundles
                ),
                "score_runtime": ARCHIVED_FAMILY_ENSEMBLE_ARTIFACT,
                "head_name": ",".join(family for family in ARCHIVED_FAMILY_ORDER if family in bundles),
            }
            continue
        if is_legacy_family_bundle(str(path)):
            meta = _try_json(path / "model_meta.json")
            head_files = sorted(path.glob("head_*.joblib"))
            head_name = head_files[0].stem.removeprefix("head_") if head_files else None
            found[key] = {
                "path": str(path),
                "type": LEGACY_FAMILY_ARTIFACT,
                "version": meta.get("split_mode"),
                "feature_count": meta.get("features_count"),
                "family": meta.get("family"),
                "created": meta.get("created"),
                "score_runtime": LEGACY_FAMILY_ARTIFACT,
                "head_name": head_name,
            }
    return sorted(found.values(), key=lambda item: (item.get("type") or "", item["path"]))


def resolve_model_directory(models_dir: str) -> str:
    model_path = Path(models_dir)
    if (
        is_multi_head_release(str(model_path))
        or is_scenario_lgbm_bundle(str(model_path))
        or is_archived_family_ensemble(str(model_path))
        or has_legacy_simple_runtime(str(model_path))
        or is_legacy_family_bundle(str(model_path))
    ):
        return str(model_path)
    discovered = discover_model_bundles(models_dir)
    if not discovered:
        raise FileNotFoundError(f"No supported model bundle found in {models_dir}")
    family_keys = {(item.get("type"), item.get("family")) for item in discovered}
    if len(discovered) > 1 and len(family_keys) == 1:
        picked = max(discovered, key=lambda item: (str(item.get("created") or ""), item["path"]))
        return picked["path"]
    if len(discovered) == 1:
        return discovered[0]["path"]
    preview = "\n".join(f"  - {item['path']}" for item in discovered[:20])
    raise ValueError(
        f"{models_dir} contains multiple model bundles. Use an exact bundle path:\n{preview}"
    )


def load_runtime(models_dir: str, *, allow_legacy: bool = True) -> Any:
    resolved_dir = resolve_model_directory(models_dir)
    if is_multi_head_release(resolved_dir):
        return RSUMultiHeadRuntime.load(resolved_dir)
    if is_scenario_lgbm_bundle(resolved_dir):
        return ScenarioLGBMRuntime(resolved_dir)
    if is_archived_family_ensemble(resolved_dir):
        return ArchivedFamilyEnsembleRuntime(resolved_dir)
    if allow_legacy and has_legacy_simple_runtime(resolved_dir):
        return LegacySimpleRuntime(resolved_dir)
    if is_legacy_family_bundle(resolved_dir):
        return LegacyFamilyRuntime(resolved_dir)
    raise FileNotFoundError(f"No supported model bundle found in {models_dir}")


def list_model_directories(root_dir: str) -> List[Dict[str, Any]]:
    return discover_model_bundles(root_dir)


def _load_joblib_compat(path: Path) -> Any:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Trying to unpickle estimator.*")
        try:
            return joblib.load(path)
        except ModuleNotFoundError as exc:
            missing = getattr(exc, "name", "") or "a required package"
            if missing in {"lightgbm", "tensorflow"}:
                raise ModuleNotFoundError(_missing_dependency_message(missing, f"loading {path}")) from exc
            raise


def _coerce_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric
    codes, _ = pd.factorize(series.astype(str), sort=True)
    return pd.Series(codes, index=series.index, dtype=float)


def _rolling_sender_mean(df: pd.DataFrame, source_col: str, target_col: str, window_size: int) -> None:
    sender = df.get("sender_pseudo", df.get("sender"))
    if sender is None or source_col not in df.columns:
        df[target_col] = 0.0
        return
    df[target_col] = (
        pd.to_numeric(df[source_col], errors="coerce")
        .fillna(0.0)
        .groupby(sender.astype(str), sort=False)
        .transform(lambda s: s.rolling(window_size, min_periods=1).mean())
    )


def list_historical_results(root_dir: str) -> List[Dict[str, Any]]:
    root_path = Path(root_dir)
    if not root_path.exists():
        return []

    def display_date_for(path: Path) -> str:
        if "results and models" in str(path):
            return "2025-12-20"
        return time.strftime("%Y-%m-%d", time.localtime(path.stat().st_mtime))

    def rel_path_for(path: Path) -> str:
        try:
            return str(path.relative_to(root_path))
        except Exception:
            return str(path)

    results: List[Dict[str, Any]] = []
    for path in sorted(root_path.rglob("*")):
        if not path.is_file():
            continue
        name = path.name
        suffix = path.suffix.lower()
        item: Optional[Dict[str, Any]] = None
        if name == "training_report.json":
            report = _try_json(path)
            item = {
                "path": str(path),
                "relative_path": rel_path_for(path),
                "display_date": display_date_for(path),
                "type": "training_report",
                "rows": report.get("rows"),
                "threshold": report.get("threshold"),
                "score": report.get("roc_auc"),
                "metric_name": "roc_auc",
                "note": ",".join(report.get("enabled_heads", [])),
            }
        elif name == "verify_summary.json":
            report = _try_json(path)
            metrics = report.get("classification_report", {})
            attack_metrics = metrics.get("1", {})
            weighted = metrics.get("weighted avg", {})
            attack_support = attack_metrics.get("support")
            attack_f1 = attack_metrics.get("f1-score")
            weighted_f1 = weighted.get("f1-score")
            use_attack_f1 = attack_f1 is not None and float(attack_support or 0) > 0
            item = {
                "path": str(path),
                "relative_path": rel_path_for(path),
                "display_date": display_date_for(path),
                "type": "verify_summary",
                "rows": report.get("rows"),
                "threshold": None,
                "score": attack_f1 if use_attack_f1 else weighted_f1,
                "metric_name": "attack_f1" if use_attack_f1 else "weighted_f1",
                "note": f"attack_f1; attack_support={attack_support}" if use_attack_f1 else "weighted_f1",
            }
        elif name.startswith("eval_report_") and suffix == ".json":
            report = _try_json(path)
            model_metrics = report.get("model_metrics", {})
            attack_id_text = name.removeprefix("eval_report_").removesuffix(".json")
            attack_id = int(attack_id_text) if attack_id_text.isdigit() else None
            item = {
                "path": str(path),
                "relative_path": rel_path_for(path),
                "display_date": display_date_for(path),
                "type": "legacy_eval_report",
                "rows": report.get("rows_eval_dt<=60.0"),
                "threshold": None,
                "score": model_metrics.get("f1"),
                "metric_name": "model_f1",
                "note": (
                    f"attack_id={attack_id} {ATTACK_TYPES.get(attack_id, 'unknown')}, "
                    f"attack_support={model_metrics.get('support_attacks')}"
                ),
            }
        elif name == "summary_eval_existing.csv":
            try:
                df = pd.read_csv(path, low_memory=False)
                item = {
                    "path": str(path),
                    "relative_path": rel_path_for(path),
                    "display_date": display_date_for(path),
                    "type": "legacy_eval_summary_csv",
                    "rows": int(len(df)),
                    "threshold": None,
                    "score": float(pd.to_numeric(df.get("Model_F1"), errors='coerce').mean()),
                    "metric_name": "mean_model_f1",
                    "note": "mean_Model_F1",
                }
            except Exception:
                item = {
                    "path": str(path),
                    "relative_path": rel_path_for(path),
                    "display_date": display_date_for(path),
                    "type": "legacy_eval_summary_csv",
                    "rows": None,
                    "threshold": None,
                    "score": None,
                    "metric_name": "mean_model_f1",
                    "note": "",
                }
        elif name.endswith("_family_metrics.csv"):
            try:
                df = pd.read_csv(path, low_memory=False)
                score_col = "F1" if "F1" in df.columns else None
                mean_score = float(pd.to_numeric(df[score_col], errors="coerce").mean()) if score_col else None
                best_label = ""
                if score_col and "Attack family" in df.columns and not df.empty:
                    best_idx = pd.to_numeric(df[score_col], errors="coerce").fillna(-1.0).idxmax()
                    best_row = df.loc[best_idx]
                    best_label = f"best={best_row.get('Attack family', 'unknown')}:{best_row.get(score_col)}"
                item = {
                    "path": str(path),
                    "relative_path": rel_path_for(path),
                    "display_date": display_date_for(path),
                    "type": "family_metrics_csv",
                    "rows": int(len(df)),
                    "threshold": None,
                    "score": mean_score,
                    "metric_name": "mean_f1",
                    "note": best_label or "mean_F1",
                }
            except Exception:
                item = {
                    "path": str(path),
                    "relative_path": rel_path_for(path),
                    "display_date": display_date_for(path),
                    "type": "family_metrics_csv",
                    "rows": None,
                    "threshold": None,
                    "score": None,
                    "metric_name": "mean_f1",
                    "note": "",
                }
        elif name.endswith("_cleaning_stats.csv"):
            item = {
                "path": str(path),
                "relative_path": rel_path_for(path),
                "display_date": display_date_for(path),
                "type": "cleaning_stats_csv",
                "rows": None,
                "threshold": None,
                "score": None,
                "metric_name": None,
                "note": "data cleaning summary",
            }
        elif name.endswith("_class_distribution.csv"):
            item = {
                "path": str(path),
                "relative_path": rel_path_for(path),
                "display_date": display_date_for(path),
                "type": "class_distribution_csv",
                "rows": None,
                "threshold": None,
                "score": None,
                "metric_name": None,
                "note": "class distribution",
            }
        elif name.endswith("_scaling_stats.csv"):
            item = {
                "path": str(path),
                "relative_path": rel_path_for(path),
                "display_date": display_date_for(path),
                "type": "scaling_stats_csv",
                "rows": None,
                "threshold": None,
                "score": None,
                "metric_name": None,
                "note": "feature scaling summary",
            }
        if item is not None:
            results.append(item)
    return sorted(
        results,
        key=lambda item: (
            item.get("display_date") or "",
            float(item.get("score")) if isinstance(item.get("score"), (int, float)) else -1.0,
            item.get("relative_path") or item["path"],
        ),
        reverse=True,
    )


class LegacyFamilyRuntime:
    """Compatibility runtime for archived family-model bundles with model_meta.json."""

    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        self.model_meta = _load_json(self.models_dir / "model_meta.json")
        self.family = str(self.model_meta.get("family", "unknown"))
        self.created = self.model_meta.get("created")
        self.preproc = _load_joblib_compat(self.models_dir / "preproc.joblib")
        self.bin_calib = _load_joblib_compat(self.models_dir / "bin_calib.joblib")
        self.meta_model = _load_joblib_compat(self.models_dir / "meta.joblib")
        head_files = sorted(self.models_dir.glob("head_*.joblib"))
        self.head_path = head_files[0] if head_files else None
        self.head_name = self.head_path.stem.removeprefix("head_") if self.head_path else self.family
        self.head_model = _load_joblib_compat(self.head_path) if self.head_path else None
        self.feature_columns = list(self.preproc.get("X_cols_final") or self.preproc.get("tfm", {}).get("keep_cols", []))
        self.preproc_tfm = dict(self.preproc.get("tfm", {}))
        self.medians = dict(self.preproc_tfm.get("medians", {}))
        self.scaler = self.preproc.get("scaler")
        self.window_size = 25 if any("_w25" in col for col in self.feature_columns) else 15

    def _build_feature_candidates(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        normalized_df = normalize_bsm_dataframe(raw_df)
        prepared_df, _ = prepare_thesis_dataframe(
            raw_df,
            window_size=self.window_size,
            obu_thresholds=thesis_obu_thresholds(),
        )
        out = prepared_df.copy()
        for col in normalized_df.columns:
            if col not in out.columns:
                out[col] = normalized_df[col].to_numpy()

        heading = pd.to_numeric(out.get("heading_curr"), errors="coerce").fillna(0.0)
        speed = pd.to_numeric(out.get("speed_curr"), errors="coerce").fillna(0.0)
        accel = pd.to_numeric(out.get("acc_curr"), errors="coerce").fillna(0.0)
        sender_codes = pd.factorize(out.get("sender_pseudo", pd.Series("", index=out.index)).astype(str), sort=True)[0]
        receiver_codes = pd.factorize(out.get("receiver_pseudo", pd.Series("", index=out.index)).astype(str), sort=True)[0]

        out["sender"] = sender_codes.astype(float)
        out["receiver_pseudo"] = receiver_codes.astype(float)
        out["window_id"] = (
            out.groupby(out.get("sender_pseudo", pd.Series("", index=out.index)).astype(str), sort=False).cumcount() // max(self.window_size, 1)
        ).astype(float)
        out["spdx"] = speed * np.cos(heading)
        out["spdy"] = speed * np.sin(heading)
        out["spdx_n"] = out["spdx"]
        out["spdy_n"] = out["spdy"]
        out["hedx"] = np.cos(heading)
        out["hedy"] = np.sin(heading)
        out["hedx_n"] = out["hedx"]
        out["hedy_n"] = out["hedy"]
        out["aclx"] = accel * np.cos(heading)
        out["acly"] = accel * np.sin(heading)
        out["aclx_n"] = out["aclx"]
        out["acly_n"] = out["acly"]
        out["sin_a"] = np.sin(heading)
        out["cos_a"] = np.cos(heading)
        out["dr_dx"] = pd.to_numeric(out.get("dx"), errors="coerce").fillna(0.0)
        out["dr_dy"] = pd.to_numeric(out.get("dy"), errors="coerce").fillna(0.0)
        out["dr_angle"] = pd.to_numeric(out.get("dtheta"), errors="coerce").fillna(0.0)
        _rolling_sender_mean(out, "dr_angle", "dr_angle_var_w25", self.window_size)
        out["dr_angle_var_w25"] = (
            pd.to_numeric(out["dr_angle_var_w25"], errors="coerce").fillna(0.0)
            .groupby(out.get("sender_pseudo", pd.Series("", index=out.index)).astype(str), sort=False)
            .transform(lambda s: s.rolling(self.window_size, min_periods=1).var().fillna(0.0))
        )
        out["posx_n"] = pd.to_numeric(out.get("x_curr"), errors="coerce").fillna(0.0)
        out["posy_n"] = pd.to_numeric(out.get("y_curr"), errors="coerce").fillna(0.0)
        out["low_speed_flag"] = (speed <= 1.0).astype(float)
        out["neg_acc_flag"] = (accel < 0.0).astype(float)
        _rolling_sender_mean(out, "low_speed_flag", f"low_speed_ratio_w{self.window_size}", self.window_size)
        _rolling_sender_mean(out, "neg_acc_flag", f"neg_acc_ratio_w{self.window_size}", self.window_size)
        dt_mean = pd.to_numeric(out.get(f"dt_mean_w{self.window_size}"), errors="coerce").fillna(0.0)
        dt_std = pd.to_numeric(out.get(f"dt_std_w{self.window_size}"), errors="coerce").fillna(0.0)
        dt = pd.to_numeric(out.get("dt"), errors="coerce").fillna(0.0)
        out["dt_cv_w"] = dt_std / (dt_mean.abs() + 1e-9)
        out["dt_z"] = (dt - dt_mean) / (dt_std.abs() + 1e-9)
        out["state_code"] = _coerce_numeric(out.get("state_code", pd.Series(0, index=out.index)))
        out["messageID"] = _coerce_numeric(out.get("messageID", pd.Series(0, index=out.index)))
        out["class"] = _coerce_numeric(out.get("class", pd.Series(0, index=out.index)))
        out["state_dup_ratio_w"] = 0.0
        return out

    def _build_model_matrix(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        candidates = self._build_feature_candidates(raw_df)
        X = pd.DataFrame(index=candidates.index)
        for col in self.feature_columns:
            if col in candidates.columns:
                series = candidates[col]
            elif col in self.medians:
                series = pd.Series(self.medians[col], index=candidates.index)
            else:
                series = pd.Series(0.0, index=candidates.index)
            numeric = _coerce_numeric(pd.Series(series, index=candidates.index))
            fill_value = self.medians.get(col, 0.0)
            X[col] = numeric.replace([np.inf, -np.inf], np.nan).fillna(fill_value)
        return X.astype(float)

    def score_components(self, raw_df: pd.DataFrame) -> Dict[str, Any]:
        X = self._build_model_matrix(raw_df)
        if self.scaler is not None:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                index=X.index,
                columns=self.feature_columns,
            )
        else:
            X_scaled = X.copy()

        p_binary = self.bin_calib.predict_proba(X_scaled)[:, 1]
        p_head = self.head_model.predict_proba(X_scaled)[:, 1] if self.head_model is not None else p_binary

        meta_width = int(getattr(self.meta_model, "n_features_in_", 2))
        if meta_width <= 1:
            meta_input = p_head.reshape(-1, 1)
        else:
            meta_input = np.column_stack([p_binary, p_head])
            if meta_width > meta_input.shape[1]:
                pad = np.zeros((len(meta_input), meta_width - meta_input.shape[1]), dtype=float)
                meta_input = np.hstack([meta_input, pad])

        p_final = self.meta_model.predict_proba(meta_input)[:, 1]
        return {
            "family": self.family,
            "head_name": self.head_name,
            "created": self.created,
            "p_binary": p_binary,
            "p_head": p_head,
            "p_final": p_final,
            "meta_width": meta_width,
            "feature_count": len(self.feature_columns),
            "has_explicit_head": self.head_model is not None,
        }

    def score_dataframe(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        base_df = normalize_bsm_dataframe(raw_df)
        components = self.score_components(raw_df)
        p_binary = components["p_binary"]
        p_head = components["p_head"]
        p_final = components["p_final"]
        out = base_df.copy()
        out["p_binary_legacy"] = p_binary
        out[f"p_{self.head_name}"] = p_head
        out["p_final"] = p_final
        out["final_decision"] = (p_final >= 0.5).astype(int)
        out["legacy_family"] = self.family
        out["legacy_head_name"] = self.head_name
        out["legacy_created"] = self.created
        out["runtime_mode"] = LEGACY_FAMILY_ARTIFACT
        return out


class ArchivedFamilyEnsembleRuntime:
    """Compatibility runtime that combines archived per-family bundles into the offline IDS flow."""

    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        self.bundle_paths = latest_archived_family_bundles(models_dir)
        missing = [family for family in ARCHIVED_FAMILY_ORDER if family not in self.bundle_paths]
        if missing:
            raise FileNotFoundError(f"Archived family model collection is missing families: {missing}")
        self.family_runtimes = {
            family: LegacyFamilyRuntime(path)
            for family, path in self.bundle_paths.items()
        }
        self.trust_config = {
            "base_threshold": 0.370,
            "sensitivity": 0.4,
            "floor": 0.35,
            "ceil": 0.85,
            "w_bad": 1.0,
            "w_good": 0.5,
            "w_bad_minor": 0.2,
        }
        self.manifest = {
            "artifact_family": ARCHIVED_FAMILY_ENSEMBLE_ARTIFACT,
            "artifact_version": "1.0.0",
            "thesis_grade_complete": False,
            "chapter3_operations_applied": True,
            "enabled_heads": list(HEAD_VECTOR_ORDER),
            "source_root": str(self.models_dir),
            "family_bundles": {
                family: {
                    "path": path,
                    "created": self.family_runtimes[family].created,
                    "feature_count": len(self.family_runtimes[family].feature_columns),
                    "has_explicit_head": self.family_runtimes[family].head_model is not None,
                }
                for family, path in self.bundle_paths.items()
            },
            "compatibility_note": (
                "Archived per-family bundles are fused into the offline IDS steps. "
                "This is not a strict rsu_multi_head_v3 release because replay_stale, DoS, "
                "and Sybil are stored as calibrated family binaries rather than one saved global bundle."
            ),
        }

    @staticmethod
    def _clip_score(values: Any) -> np.ndarray:
        return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)

    @staticmethod
    def _fusion_contribution(values: Any) -> np.ndarray:
        scores = ArchivedFamilyEnsembleRuntime._clip_score(values)
        if scores.size == 0:
            return scores
        q50 = float(np.quantile(scores, 0.50))
        q05 = float(np.quantile(scores, 0.05))
        q95 = float(np.quantile(scores, 0.95))
        max_score = float(np.max(scores))
        if q05 >= 0.85 and q95 >= 0.95:
            return np.zeros_like(scores, dtype=float)
        if q50 < 0.10 and max_score >= 0.99:
            return (scores >= 0.99).astype(float)
        if q95 < 0.37:
            if max_score >= 0.95:
                return (scores >= 0.99).astype(float)
            return np.zeros_like(scores, dtype=float)
        return scores

    def _score_family_components(self, raw_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        components: Dict[str, Dict[str, Any]] = {}
        for family in ARCHIVED_FAMILY_ORDER:
            components[family] = self.family_runtimes[family].score_components(raw_df)
        return components

    def score_dataframe(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        # Phase 1: normalize the extracted BSM rows and compute OBU evidence flags
        # such as physical speed, acceleration, heading-rate, consistency, and timestamp checks.
        scored, _ = prepare_thesis_dataframe(
            raw_df,
            window_size=DEFAULT_WINDOW_SIZE,
            obu_thresholds=thesis_obu_thresholds(),
        )

        # Phase 2: run the archived family preprocessors and models. Each family bundle
        # rebuilds the same engineered feature space used during its original training.
        components = self._score_family_components(raw_df)

        # Phase 3: collect the RSU detector-head scores into one dataframe.
        # The explicit pos_speed head is used when available; older families use their calibrated family score.
        binary_scores = []
        for family in ARCHIVED_FAMILY_ORDER:
            comp = components[family]
            binary_scores.append(self._clip_score(comp["p_binary"]))
            family_score = comp["p_head"] if comp.get("has_explicit_head") else comp["p_final"]
            scored[f"p_{family}"] = self._clip_score(family_score)

        scored["p_general"] = np.maximum.reduce(binary_scores) if binary_scores else np.zeros(len(scored), dtype=float)
        scored["p_dos_iforest"] = scored["p_dos"].astype(float)
        scored["p_integrity"] = np.maximum(
            scored["obu_risk"].astype(float),
            scored["flag_consistency"].astype(float),
        )

        # Phase 4: fuse the detector heads into p_final. Saturated archived heads are
        # down-weighted so a model that returns almost all 1.0 values cannot mark every row as malicious.
        scored["p_final"] = np.maximum.reduce(
            [
                self._fusion_contribution(scored["p_general"]),
                self._fusion_contribution(scored["p_pos_speed"]),
                self._fusion_contribution(scored["p_replay_stale"]),
                self._fusion_contribution(scored["p_dos"]),
                self._fusion_contribution(scored["p_dos_iforest"]),
                self._fusion_contribution(scored["p_sybil"]),
                0.3 * scored["p_integrity"].to_numpy(dtype=float),
            ]
        )

        # Phase 5: apply sender-level trust gating. The threshold changes per vehicle
        # as the sender gains or loses trust across the ordered message stream.
        trust = AdaptiveTrustManager(self.trust_config)
        scored = scored.sort_values(["t_curr", "sender_pseudo", "row_id"], kind="mergesort").reset_index(drop=True)

        trust_before = []
        trust_after = []
        adaptive_thresholds = []
        final_decisions = []
        for _, row in scored.iterrows():
            sender = str(row["sender_pseudo"])
            before = trust.trust(sender)
            threshold = trust.threshold(sender)
            decision = int(float(row["p_final"]) >= threshold)
            trust.update_after_decision(sender, decision=decision, has_obu_flags=trust.has_obu_flags(row))
            after = trust.trust(sender)
            trust_before.append(before)
            adaptive_thresholds.append(threshold)
            final_decisions.append(decision)
            trust_after.append(after)

        scored["trust_sender_before"] = trust_before
        scored["adaptive_threshold"] = adaptive_thresholds
        scored["final_decision"] = final_decisions
        scored["trust_sender_after"] = trust_after
        scored["rsu_score"] = scored["p_final"].astype(float)
        scored["rsu_anom"] = scored["final_decision"].astype(int)
        scored["fused_risk"] = scored["p_final"].astype(float)
        scored["fused_anom"] = scored["final_decision"].astype(int)
        scored["runtime_mode"] = ARCHIVED_FAMILY_ENSEMBLE_ARTIFACT

        keep_order = [
            "row_id",
            "row_key",
            "receiver_pseudo",
            "sender_pseudo",
            "t_curr",
            "dt",
            "label",
            "attack_id",
            *OBU_FLAG_COLUMNS,
            "obu_flag_count",
            "obu_risk",
            "obu_anom",
            "p_general",
            "p_pos_speed",
            "p_replay_stale",
            "p_dos",
            "p_dos_iforest",
            "p_sybil",
            "p_integrity",
            "p_final",
            "adaptive_threshold",
            "trust_sender_before",
            "trust_sender_after",
            "final_decision",
            "rsu_score",
            "rsu_anom",
            "fused_risk",
            "fused_anom",
            "runtime_mode",
        ]
        available = [col for col in keep_order if col in scored.columns]
        return scored[available].copy()


def _scenario_feature_config(df: pd.DataFrame, *, family: str, window_size: int, seq_len: int) -> Dict[str, Any]:
    base_config = build_feature_config(df, window_size=window_size, seq_len=seq_len)
    if family == "pos_speed":
        feature_columns = select_pos_speed_features(base_config["feature_columns"])
    else:
        feature_columns = list(base_config["feature_columns"])
    fill_values = (
        df[feature_columns]
        .median(numeric_only=True)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_dict()
    )
    binary_feature_columns = [col for col in feature_columns if is_binary_feature_series(df[col])]
    continuous_feature_columns = [col for col in feature_columns if col not in set(binary_feature_columns)]
    return {
        "window_size": int(window_size),
        "seq_len": int(seq_len),
        "feature_columns": list(feature_columns),
        "binary_feature_columns": list(binary_feature_columns),
        "continuous_feature_columns": list(continuous_feature_columns),
        "fill_values": {key: float(value) for key, value in fill_values.items()},
        "trained_family": family,
    }


def infer_training_family(df: pd.DataFrame) -> str:
    ids = set(pd.to_numeric(df.get("attack_id", pd.Series(0, index=df.index)), errors="coerce").fillna(0).astype(int).tolist())
    ids.discard(0)
    if ids and ids <= ATTACK_FAMILIES["pos_speed"]:
        return "pos_speed"
    if ids and ids <= ATTACK_FAMILIES["replay_stale"]:
        return "replay_stale"
    if ids and ids <= ATTACK_FAMILIES["dos"]:
        return "dos"
    if ids and ids <= ATTACK_FAMILIES["sybil"]:
        return "sybil"
    return "binary"


def train_scenario_lgbm_release(
    csv_path: str,
    *,
    output_dir: str,
    family: str = "auto",
    window_size: int = DEFAULT_WINDOW_SIZE,
    seq_len: int = DEFAULT_SEQ_LEN,
) -> Dict[str, Any]:
    require_lightgbm("training a small-scenario Baghdad detector")
    raw_df = pd.read_csv(csv_path, low_memory=False)
    if "label" not in raw_df.columns:
        raise ValueError("Scenario training requires a labeled CSV with a 'label' column.")
    prepared_df, obu_thresholds = prepare_thesis_dataframe(
        raw_df,
        window_size=window_size,
        obu_thresholds=thesis_obu_thresholds(),
    )
    y = prepared_df["label"].astype(int)
    if y.nunique() < 2:
        raise ValueError("Scenario training requires both benign and attack labels.")
    trained_family = infer_training_family(prepared_df) if family == "auto" else family
    feature_config = _scenario_feature_config(
        prepared_df,
        family=trained_family,
        window_size=window_size,
        seq_len=seq_len,
    )
    X = align_feature_matrix(prepared_df, feature_config)

    indices = np.arange(len(prepared_df))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )
    params = {
        "objective": "binary",
        "class_weight": "balanced",
        "n_estimators": 120,
        "learning_rate": 0.07,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.90,
        "random_state": 42,
        "verbosity": -1,
    }
    validation_model = lgb.LGBMClassifier(**params)
    validation_model.fit(X.iloc[train_idx], y.iloc[train_idx])
    val_scores = validation_model.predict_proba(X.iloc[val_idx])[:, 1]
    threshold = best_f1_threshold(y.iloc[val_idx], val_scores, default=0.5)
    val_pred = (val_scores >= threshold).astype(int)
    val_report = classification_report(y.iloc[val_idx], val_pred, digits=4, output_dict=True, zero_division=0)
    val_cm = confusion_matrix(y.iloc[val_idx], val_pred)
    val_auc = float(roc_auc_score(y.iloc[val_idx], val_scores)) if y.iloc[val_idx].nunique() >= 2 else None

    final_model = lgb.LGBMClassifier(**params)
    final_model.fit(X, y)

    output_path = Path(ensure_dir(output_dir))
    joblib.dump(final_model, output_path / "scenario_lgbm.joblib")
    with open(output_path / "features.json", "w", encoding="utf-8") as handle:
        json.dump(json_ready(feature_config), handle, indent=2)
    with open(output_path / "obu_thresholds.json", "w", encoding="utf-8") as handle:
        json.dump(json_ready(obu_thresholds), handle, indent=2)
    manifest = {
        "artifact_family": SCENARIO_LGBM_ARTIFACT,
        "artifact_version": "1.0.0",
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_name": csv_path,
        "feature_count": int(len(feature_config["feature_columns"])),
        "trained_family": trained_family,
        "enabled_heads": ["general"] + ([] if trained_family == "binary" else [trained_family]),
        "threshold": float(threshold),
        "training_protocol": "small_scenario_stratified_holdout",
        "warning": "Use for small labeled Baghdad/F2MD scenario validation; not a replacement for the full sender-disjoint multi-head release.",
    }
    with open(output_path / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(json_ready(manifest), handle, indent=2)
    report = {
        **manifest,
        "rows": int(len(prepared_df)),
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "label_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "validation_roc_auc": val_auc,
        "validation_confusion_matrix": json_ready(val_cm),
        "validation_classification_report": json_ready(val_report),
    }
    with open(output_path / "training_report.json", "w", encoding="utf-8") as handle:
        json.dump(json_ready(report), handle, indent=2)
    return report


class ScenarioLGBMRuntime:
    """Small-scenario detector using the thesis feature pipeline and a calibrated LightGBM threshold."""

    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        self.manifest = _load_json(self.models_dir / "manifest.json")
        self.feature_config = _load_json(self.models_dir / "features.json")
        self.obu_thresholds = _load_json(self.models_dir / "obu_thresholds.json")
        self.model = _load_joblib_compat(self.models_dir / "scenario_lgbm.joblib")
        self.threshold = float(self.manifest.get("threshold", 0.5))
        self.trained_family = str(self.manifest.get("trained_family", "binary"))

    def score_dataframe(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        prepared_df, _ = prepare_thesis_dataframe(
            raw_df,
            window_size=int(self.feature_config.get("window_size", DEFAULT_WINDOW_SIZE)),
            obu_thresholds=self.obu_thresholds or thesis_obu_thresholds(),
        )
        X = align_feature_matrix(prepared_df, self.feature_config)
        scores = self.model.predict_proba(X)[:, 1]
        out = prepared_df.copy()
        out["p_general"] = scores
        if self.trained_family != "binary":
            out[f"p_{self.trained_family}"] = scores
        out["p_final"] = scores
        out["adaptive_threshold"] = self.threshold
        out["final_decision"] = (scores >= self.threshold).astype(int)
        out["rsu_score"] = out["p_final"].astype(float)
        out["rsu_anom"] = out["final_decision"].astype(int)
        out["fused_risk"] = out["p_final"].astype(float)
        out["fused_anom"] = out["final_decision"].astype(int)
        out["runtime_mode"] = SCENARIO_LGBM_ARTIFACT
        keep_order = [
            "row_id",
            "row_key",
            "receiver_pseudo",
            "sender_pseudo",
            "t_curr",
            "dt",
            "label",
            "attack_id",
            *OBU_FLAG_COLUMNS,
            "obu_flag_count",
            "obu_risk",
            "obu_anom",
            "p_general",
            f"p_{self.trained_family}",
            "p_final",
            "adaptive_threshold",
            "final_decision",
            "rsu_score",
            "rsu_anom",
            "fused_risk",
            "fused_anom",
            "runtime_mode",
        ]
        available = [col for col in keep_order if col in out.columns]
        return out[available].copy()


class LegacySimpleRuntime:
    """Explicit fallback for old RandomForest/IsolationForest runtime bundles."""

    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        config_path = self.models_dir / "fusion_config_final_v2.json"
        with open(config_path, "r", encoding="utf-8") as handle:
            self.config = json.load(handle)
        self.params = self.config.get("params", {})
        with open(self.models_dir / self.config["models"]["obu_thresholds"], "r", encoding="utf-8") as handle:
            self.obu_thresholds = json.load(handle)
        self.rsu_model = _load_joblib_compat(self.models_dir / self.config["models"]["rsu_supervised"])
        with open(self.models_dir / self.config["models"]["rsu_info"], "r", encoding="utf-8") as handle:
            self.rsu_info = json.load(handle)
        self.iforest_bundle = None
        if self.config["models"].get("iforest"):
            self.iforest_bundle = _load_joblib_compat(self.models_dir / self.config["models"]["iforest"])
        self.sender_trust_db = pd.read_csv(self.models_dir / self.config["artifacts"]["sender_trust"])
        self.pair_trust_db = pd.read_csv(self.models_dir / self.config["artifacts"]["pair_trust"])

    def _compute_legacy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = normalize_bsm_dataframe(df)
        out["delta_speed"] = out["speed_curr"] - out["speed_prev"]
        denom = out["dt"].replace(0.0, np.nan)
        out["jerk"] = ((out["acc_curr"] - out["acc_prev"]) / denom).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        out["heading_rate"] = (out["dtheta"] / denom).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        if self.iforest_bundle is not None:
            use_feats = self.iforest_bundle["features"]
            X_if = out.reindex(columns=use_feats, fill_value=0.0).fillna(0.0).to_numpy(dtype=float)
            score_normal = self.iforest_bundle["model"].decision_function(X_if)
            out["anom_score_iforest"] = 1.0 - minmax01(score_normal)
        else:
            out["anom_score_iforest"] = 0.0
        return out

    def score_dataframe(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = self._compute_legacy_features(raw_df)
        thr = self.obu_thresholds
        dt_ok = df["dt"] <= float(thr.get("dt_max_used", 2.0))
        delta_speed = df["delta_speed"]
        v_jerk = df["jerk"].abs() > float(thr.get("jerk_abs_max", 1e9))
        v_hr = df["heading_rate"].abs() > float(thr.get("heading_rate_abs_max", 1e9))
        v_ds = delta_speed.abs() > float(thr.get("delta_speed_abs_max", 1e9))
        df["obu_anom"] = ((dt_ok & v_jerk) | (dt_ok & v_hr) | (dt_ok & v_ds)).astype(int)
        df["obu_risk"] = ((dt_ok & v_jerk).astype(int) + (dt_ok & v_hr).astype(int) + (dt_ok & v_ds).astype(int)) / 3.0

        feature_names = self.rsu_info["features"]
        X = df.reindex(columns=feature_names, fill_value=0.0).fillna(0.0).to_numpy(dtype=float)
        rsu_score = self.rsu_model.predict_proba(X)[:, 1]
        rsu_thr = float(self.params.get("rsu_thr", 0.5))
        rsu_soft = float(self.params.get("rsu_soft", 0.35))
        sender_cut = float(self.params.get("tsender_cut", 0.965))
        pair_cut = float(self.params.get("tpair_cut", 0.85))

        df["sender_pseudo"] = df["sender_pseudo"].astype(str)
        df["receiver_pseudo"] = df["receiver_pseudo"].astype(str)
        sender_map = self.sender_trust_db[["sender_pseudo", "trust_blend"]].rename(columns={"trust_blend": "trust_sender"}).copy()
        pair_map = self.pair_trust_db[["receiver_pseudo", "sender_pseudo", "trust_blend"]].rename(columns={"trust_blend": "trust_pair"}).copy()
        sender_map["sender_pseudo"] = sender_map["sender_pseudo"].astype(str)
        pair_map["sender_pseudo"] = pair_map["sender_pseudo"].astype(str)
        pair_map["receiver_pseudo"] = pair_map["receiver_pseudo"].astype(str)
        merged = df.merge(sender_map, on="sender_pseudo", how="left").merge(
            pair_map,
            on=["receiver_pseudo", "sender_pseudo"],
            how="left",
        )
        merged["trust_sender"] = merged["trust_sender"].fillna(0.5)
        merged["trust_pair"] = merged["trust_pair"].fillna(0.5)
        merged["rsu_score"] = rsu_score
        merged["rsu_anom"] = (merged["rsu_score"] >= rsu_thr).astype(int)
        rsu_hit = merged["rsu_score"] >= rsu_soft
        high_trust = (merged["trust_sender"] >= sender_cut) | (merged["trust_pair"] >= pair_cut)
        obu_only = (merged["obu_anom"] == 1) & (~rsu_hit)
        suppress = obu_only & high_trust
        merged["final_decision"] = (rsu_hit | ((merged["obu_anom"] == 1) & (~suppress))).astype(int)
        merged["p_final"] = merged["rsu_score"].astype(float)
        merged["adaptive_threshold"] = rsu_thr
        merged["trust_sender_before"] = merged["trust_sender"].astype(float)
        merged["trust_sender_after"] = merged["trust_sender"].astype(float)
        merged["fused_anom"] = merged["final_decision"]
        merged["fused_risk"] = merged["p_final"]
        merged["runtime_mode"] = "legacy_simple_runtime"
        return merged[
            [
                "row_id",
                "row_key",
                "receiver_pseudo",
                "sender_pseudo",
                "t_curr",
                "dt",
                "label",
                "attack_id",
                "obu_anom",
                "obu_risk",
                "rsu_score",
                "rsu_anom",
                "adaptive_threshold",
                "trust_sender_before",
                "trust_sender_after",
                "p_final",
                "final_decision",
                "fused_risk",
                "fused_anom",
                "runtime_mode",
            ]
        ].copy()
