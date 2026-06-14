from __future__ import annotations

import numpy as np


def summarize_numeric_values(values: list[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "zero_count": 0,
            "zero_fraction": 0.0,
            "lt_1e-12_fraction": 0.0,
            "lt_1e-6_fraction": 0.0,
            "lt_1e-3_fraction": 0.0,
            "lt_1_fraction": 0.0,
            "min": 0.0,
            "mean": 0.0,
            "50%": 0.0,
            "95%": 0.0,
            "99%": 0.0,
            "max": 0.0,
        }

    abs_array = np.abs(array)
    return {
        "count": int(array.size),
        "zero_count": int(np.sum(array == 0.0)),
        "zero_fraction": float(np.mean(array == 0.0)),
        "lt_1e-12_fraction": float(np.mean(abs_array < 1e-12)),
        "lt_1e-6_fraction": float(np.mean(abs_array < 1e-6)),
        "lt_1e-3_fraction": float(np.mean(abs_array < 1e-3)),
        "lt_1_fraction": float(np.mean(abs_array < 1.0)),
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "50%": float(np.percentile(array, 50)),
        "95%": float(np.percentile(array, 95)),
        "99%": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def qerror_summary(predictions: list[float] | np.ndarray, targets: list[float] | np.ndarray) -> dict[str, float]:
    y_pred = np.asarray(predictions, dtype=np.float64)
    y_true = np.asarray(targets, dtype=np.float64)
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Prediction shape {y_pred.shape} does not match target shape {y_true.shape}")

    if y_pred.size == 0:
        return {"mean": 0.0, "50%": 0.0, "90%": 0.0, "95%": 0.0, "99%": 0.0, "max": 0.0}

    safe_pred = np.clip(y_pred, a_min=0.0, a_max=None)
    safe_true = np.clip(y_true, a_min=1e-6, a_max=None)
    qerrors = np.maximum(safe_true / (safe_pred + 1e-6), safe_pred / safe_true)
    return {
        "mean": float(np.mean(qerrors)),
        "50%": float(np.percentile(qerrors, 50)),
        "90%": float(np.percentile(qerrors, 90)),
        "95%": float(np.percentile(qerrors, 95)),
        "99%": float(np.percentile(qerrors, 99)),
        "max": float(np.max(qerrors)),
    }
