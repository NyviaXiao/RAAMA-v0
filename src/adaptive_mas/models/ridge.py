"""Fold-local standardized ridge regression for the five-session baseline."""

import numpy as np


def ridge_predictions(
    training_features: list[tuple[float, ...]],
    training_returns: list[float],
    prediction_features: dict[str, tuple[float, ...]],
    alpha: float,
) -> dict[str, float]:
    if len(training_features) != len(training_returns):
        raise ValueError("Ridge 训练特征与收益标签数量不一致")
    if not training_features or not prediction_features:
        return {}

    train = np.asarray(training_features, dtype=float)
    target = np.asarray(training_returns, dtype=float)
    symbols = sorted(prediction_features)
    predict = np.asarray([prediction_features[symbol] for symbol in symbols], dtype=float)
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale == 0] = 1.0
    train = (train - mean) / scale
    predict = (predict - mean) / scale
    train = np.column_stack((np.ones(len(train)), train))
    predict = np.column_stack((np.ones(len(predict)), predict))
    penalty = np.eye(train.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(train.T @ train + penalty, train.T @ target)
    values = predict @ coefficients
    return {symbol: float(value) for symbol, value in zip(symbols, values, strict=True)}
